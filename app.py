import os
import time
import datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import io, requests



@st.cache_data(ttl=300)
def load_report_by_key(key: str) -> pd.DataFrame:
    """
    Load a report by logical key: 'all' | 'breakouts' | 'breakdowns' | 'alerts'.
    1) If Streamlit secrets contain Drive file IDs, fetch from Google Drive.
    2) Else, fallback to local /alert/*.csv in the repo.
    Normalizes headers and ensures a 'Symbol' column when possible.
    """
    key = key.lower()
    drive_cfg = st.secrets.get("drive", None)

    local_map = {
        "all": "rectangle_all.csv",
        "breakouts": "rectangle_breakouts.csv",
        "breakdowns": "rectangle_breakdowns.csv",
        "alerts": "rectangle_alerts.csv",
    }

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        # strip BOM/whitespace and lower for matching
        new_cols = []
        for c in df.columns:
            if isinstance(c, str):
                c2 = c.encode("utf-8").decode("utf-8-sig").strip()
            else:
                c2 = c
            new_cols.append(c2)
        df.columns = new_cols

        # If there is an obvious index column, drop it
        for cand in ["index", "Unnamed: 0", "", None]:
            if cand in df.columns:
                # only drop if it looks like a simple 0..N integer index
                if pd.api.types.is_integer_dtype(df[cand]) or df[cand].astype(str).str.match(r"^\d+$").all():
                    df = df.drop(columns=[cand], errors="ignore")

        # Unify typical variants to 'Symbol'
        colmap = {c: c for c in df.columns}
        lower = {c.lower(): c for c in df.columns if isinstance(c, str)}
        for possible in ["symbol", "ticker", "scrip", "stock", "nse_symbol"]:
            if possible in lower:
                colmap[lower[possible]] = "Symbol"
                break

        df = df.rename(columns=colmap)

        # If still no 'Symbol' but the first column looks like symbols, use it
        if "Symbol" not in df.columns and len(df.columns) > 0:
            first = df.columns[0]
            # heuristic: short strings, letters/numbers/.- only
            sample = df[first].astype(str).head(10)
            if sample.str.match(r"^[A-Za-z0-9\.\-\_]{2,20}$").all():
                df = df.rename(columns={first: "Symbol"})

        # Coerce DATE to datetime if present
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

        return df

    # Try Google Drive first
    if drive_cfg and key in drive_cfg and "base" in drive_cfg:
        file_id = drive_cfg[key]
        base = drive_cfg["base"].rstrip()
        # Support both "https://drive.google.com/uc?id=" and "…/uc?id"
        url = base if base.endswith("=") else (base + ("&id=" if "?" in base else "?id="))
        url = f"{url}{file_id}&export=download" if "export=download" not in url else f"{url}{file_id}"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            return _normalize(df)
        except Exception as e:
            st.warning(f"Drive load failed for '{key}' ({e}). Falling back to local file.")

    # Fallback: local /alert
    local_name = local_map.get(key)
    if local_name:
        local_path = os.path.join("alert", local_name)
        if os.path.exists(local_path):
            try:
                df = pd.read_csv(local_path)
                return _normalize(df)
            except Exception as e:
                st.error(f"Local CSV read failed for {local_name}: {e}")
                return pd.DataFrame()

    st.error(f"No source found for report key '{key}'. Check secrets or local files.")
    return pd.DataFrame()


def plot_symbol_with_sr(
    symbol: str,
    supports: List[float],
    resistances: List[float],
    months: int = 6,
) -> None:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30 * months)

    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
    except Exception as e:
        st.error(f"Failed to download data for {symbol}: {e}")
        return

    if data is None or data.empty:
        st.warning(f"No historical data found for {symbol}.")
        return

    # ---- Normalize columns (yfinance may return MultiIndex) ----
    data = data.copy()
    data.index = pd.to_datetime(data.index)

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = data.columns.get_level_values(0)
        except Exception:
            try:
                data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
                data = data.loc[:, (symbol, ["Open", "High", "Low", "Close", "Volume"])]
                data.columns = ["Open", "High", "Low", "Close", "Volume"]
            except Exception as ee:
                st.error(f"Could not normalize OHLC columns for {symbol}: {ee}")
                return

    for col in ["Open", "High", "Low", "Close"]:
        if col not in data.columns:
            st.error(f"Downloaded data is missing column '{col}'.")
            return

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if data.empty:
        st.warning("Downloaded data has no valid OHLC rows after cleaning.")
        return

    # ---- Build figure (candles top, volume bottom) ----
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.78, 0.22],
        subplot_titles=(f"{symbol} — Daily", "Volume")
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1, col=1
    )

    if "Volume" in data.columns:
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], name="Volume", opacity=0.3),
            row=2, col=1
        )

    # ---- Support/Resistance (solid yellow lines + labels) ----
    def valid_levels(levels):
        out = []
        for v in levels or []:
            try:
                f = float(v)
                if not np.isnan(f):
                    out.append(f)
            except Exception:
                pass
        return sorted(set(out))

    sup_levels = valid_levels(supports)
    res_levels = valid_levels(resistances)

    all_levels = sup_levels + res_levels
    if all_levels:
        y_lo = min(float(data["Low"].min()), min(all_levels))
        y_hi = max(float(data["High"].max()), max(all_levels))
        pad = max(1e-6, 0.01 * (y_hi - y_lo))
        fig.update_yaxes(range=[y_lo - pad, y_hi + pad], row=1, col=1)

    def add_level(lvl: float, label: str):
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y1",   y0=lvl, y1=lvl,
            line=dict(color="yellow", width=1.5)  # solid yellow line
        )
        fig.add_annotation(
            x=1.0, xref="paper", xanchor="left",
            y=lvl, yref="y1",
            text=f"{label} {lvl:.2f}",
            showarrow=False, font=dict(size=10), xshift=8
        )

    for lvl in sup_levels:
        add_level(lvl, "Support")
    for lvl in res_levels:
        add_level(lvl, "Resistance")

    fig.update_layout(
        height=650,
        margin=dict(l=60, r=120, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# App
# -------------------------------
def main() -> None:
    st.set_page_config(page_title="Rectangle Pattern Reports", layout="wide")
    st.title("📊 Rectangle Pattern Reports Viewer")

    # Sidebar: report selection
    st.sidebar.header("Report Selection & Filters")
    report_type = st.sidebar.selectbox(
        "Select report",
        ["All", "Breakouts", "Breakdowns", "Alerts"],
        index=0,
        help="Choose which precomputed report to view."
    )
    report_map = {
    "All": "all",
    "Breakouts": "breakouts",
    "Breakdowns": "breakdowns",
    "Alerts": "alerts",
}
    report_key = report_map[report_type]
    df_report = load_report_by_key(report_key)

    if df_report.empty:
        st.warning("No report data found. Ensure the CSV files exist at the configured path.")
        return

    # Date conversion for filtering
    df_report["DATE"] = pd.to_datetime(df_report["DATE"], errors="coerce")

    # --- session state for double-click from table -> dropdown ---
    if "last_click_row" not in st.session_state:
        st.session_state.last_click_row = None
    if "last_click_time" not in st.session_state:
        st.session_state.last_click_time = 0.0
    if "symbol_filter" not in st.session_state:
        st.session_state.symbol_filter = "All"

    # Sidebar filter dropdown (will be controlled by double-click)
    symbols = sorted(df_report["Symbol"].dropna().unique().tolist())
    selected_symbol = st.sidebar.selectbox(
        "Filter by symbol",
        ["All"] + symbols,
        index=(["All"] + symbols).index(st.session_state.symbol_filter)
        if st.session_state.symbol_filter in (["All"] + symbols) else 0,
        key="symbol_filter",
        help="Double-click a row in the table to set this."
    )

    # Date range filter
    min_date = df_report["DATE"].min().date() if not df_report.empty else datetime.date.today()
    max_date = df_report["DATE"].max().date() if not df_report.empty else datetime.date.today()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        date_range = (min_date, max_date)

    # Apply filters
    filtered = df_report.copy()
    if selected_symbol != "All":
        filtered = filtered[filtered["Symbol"] == selected_symbol]
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["DATE"] >= pd.to_datetime(start_date)) &
        (filtered["DATE"] <= pd.to_datetime(end_date))
    ]

    st.subheader(f"{report_type} report results")
    st.write(f"Showing {len(filtered)} records out of {len(df_report)} after applying filters.")

    # ---- render table as data_editor to get selection, and detect double-click ----
    display_cols = ["Symbol", "DATE", "RESISTANCE", "SUPPORT", "DIRECTION"]
    table_df = filtered[display_cols].reset_index(drop=True).copy()
    # Freeze types for nicer display
    if "DATE" in table_df.columns:
        table_df["DATE"] = pd.to_datetime(table_df["DATE"]).dt.date.astype(str)

    st.caption("Tip: select a symbol on filter for the chart below")
    st.data_editor(
        table_df,
        key="report_table",
        hide_index=True,
        use_container_width=True,
        disabled=True,              # read-only; still allows selection
        height=min(560, 40 + 28 * max(3, len(table_df)))  # dynamic height
    )

    # Selection state from the editor
    selected_rows = []
    try:
        selected_rows = st.session_state["report_table"]["selection"]["rows"]
    except Exception:
        selected_rows = []

    if selected_rows:
        row = selected_rows[0]
        now = time.time()
        if st.session_state.last_click_row == row and (now - st.session_state.last_click_time) <= 0.7:
            # double-click detected -> update symbol filter
            try:
                sym = table_df.loc[row, "Symbol"]
                if sym in (["All"] + symbols):
                    st.session_state.symbol_filter = sym
            except Exception:
                pass
        st.session_state.last_click_row = row
        st.session_state.last_click_time = now

    # Summary metrics
    st.markdown("### Summary Metrics")
    c1, c2, c3 = st.columns(3)
    direction_col = (
        filtered["DIRECTION"].fillna("")
        if "DIRECTION" in filtered.columns
        else pd.Series(dtype=str)
    )
    with c1:
        c1.metric(
            "Breakouts",
            int(direction_col.str.contains("BREAKOUT", case=False, na=False).sum()),
        )
    with c2:
        c2.metric(
            "Breakdowns",
            int(direction_col.str.contains("BREAKDOWN", case=False, na=False).sum()),
        )
    with c3:
        c3.metric(
            "Alerts",
            int(direction_col.str.contains("ALERT", case=False, na=False).sum()),
        )

    # Chart for selected symbol with S/R
    if st.session_state.symbol_filter != "All":
        st.markdown("### Daily chart with support/resistance")
        sym_filtered = df_report[df_report["Symbol"] == st.session_state.symbol_filter]
        sup_vals = sym_filtered["SUPPORT"].dropna().tolist()
        res_vals = sym_filtered["RESISTANCE"].dropna().tolist()
        plot_symbol_with_sr(st.session_state.symbol_filter, sup_vals, res_vals)


if __name__ == "__main__":
    main()

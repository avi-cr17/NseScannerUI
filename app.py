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

    key = key.lower()
    drive_cfg = st.secrets.get("drive", None)
    local_map = {
        "all": "rectangle_all.csv",
        "breakouts": "rectangle_breakouts.csv",
        "breakdowns": "rectangle_breakdowns.csv",
        "alerts": "rectangle_alerts.csv",
    }

    # Try Google Drive (public link)
    if drive_cfg and key in drive_cfg and "base" in drive_cfg:
        file_id = drive_cfg[key]
        base = drive_cfg["base"].rstrip("?&")
        url = f"{base}={file_id}&export=download" if "uc?id" not in base else f"{base}{file_id}&export=download"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content))
        except Exception as e:
            st.warning(f"Drive load failed for '{key}' ({e}). Falling back to local file.")

    # Fallback: local /alert folder in repo
    local_name = local_map.get(key)
    if local_name:
        local_path = os.path.join("alert", local_name)
        if os.path.exists(local_path):
            try:
                return pd.read_csv(local_path, index_col=None)
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
    with c1:
        c1.metric("Breakouts", int((df_report["DIRECTION"].str.contains("BREAKOUT", case=False, na=False)).sum()))
    with c2:
        c2.metric("Breakdowns", int((df_report["DIRECTION"].str.contains("BREAKDOWN", case=False, na=False)).sum()))
    with c3:
        c3.metric("Alerts", int((df_report["DIRECTION"].str.contains("ALERT", case=False, na=False)).sum()))

    # Chart for selected symbol with S/R
    if st.session_state.symbol_filter != "All":
        st.markdown("### Daily chart with support/resistance")
        sym_filtered = df_report[df_report["Symbol"] == st.session_state.symbol_filter]
        sup_vals = sym_filtered["SUPPORT"].dropna().tolist()
        res_vals = sym_filtered["RESISTANCE"].dropna().tolist()
        plot_symbol_with_sr(st.session_state.symbol_filter, sup_vals, res_vals)


if __name__ == "__main__":
    main()


# planning_dash_improved.py
# FreshBites — Supply Planning Dashboard (Improved)
# - Adds a dedicated Safety Stock tab wired to the "Safety Stock Multiplier"
# - Adds a Strategy Formulation tab with dynamic insights
# - Keeps the rest of the functionality (overview, filtering, forecasting, simulator) similar

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional Prophet for forecasting
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="FreshBites — Supply Planning (Improved)", layout="wide")
st.title("FreshBites — Supply Planning Dashboard (Improved)")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE)  # default to script folder; change if needed

def read_default_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_data_with_upload(default_filename: str, uploaded_files):
    # If a user uploaded a file with the same name, use it; else load default
    if uploaded_files:
        for f in uploaded_files:
            if f.name == default_filename:
                return pd.read_csv(f)
    return read_default_csv(default_filename)

def safe_to_datetime(series):
    try:
        return pd.to_datetime(series)
    except Exception:
        return series

def week_to_date(week_series, base_monday="2025-01-06"):
    base = pd.to_datetime(base_monday)  # Monday of week 1 (arbitrary)
    weeks = pd.to_numeric(week_series, errors="coerce")
    return base + pd.to_timedelta(weeks - 1, unit="W")

def add_names(df, master_skus, dcs):
    df = df.copy()
    if not master_skus.empty:
        df = df.merge(master_skus[["sku_id","sku_name"]], on="sku_id", how="left")
    if not dcs.empty:
        df = df.merge(dcs[["dc_id","dc_name"]], on="dc_id", how="left")
    return df

# ---------------------------------------------------------
# Data load (with replacement via uploader)
# ---------------------------------------------------------
st.sidebar.header("Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSVs to override defaults (optional)",
    type="csv",
    accept_multiple_files=True
)

master_skus = load_data_with_upload("p1_master_skus.csv", uploaded_files)
dcs = load_data_with_upload("p1_distribution_centers.csv", uploaded_files)
inv_hist = load_data_with_upload("p1_inventory_history.csv", uploaded_files)
shortages = load_data_with_upload("p1_shortages.csv", uploaded_files)
forecasts = load_data_with_upload("p1_forecasts.csv", uploaded_files)
actuals = load_data_with_upload("p1_demand_actual.csv", uploaded_files)
inv_wk1 = load_data_with_upload("p1_inventory_wk1.csv", uploaded_files)
safety_stock = load_data_with_upload("p1_safety_stock.csv", uploaded_files)
plant_capacity = load_data_with_upload("p1_plant_capacity.csv", uploaded_files)
suppliers = load_data_with_upload("p1_supplier_data.csv", uploaded_files)

# Validation / graceful fallback
for name, df, cols in [
    ("p1_inventory_history.csv", inv_hist, ["sku_id","dc_id","week","inventory_end_units"]),
    ("p1_shortages.csv", shortages, ["sku_id","dc_id","week","shortage_units"]),
    ("p1_forecasts.csv", forecasts, ["week","sku_id","dc_id","forecast_units"]),
    ("p1_demand_actual.csv", actuals, ["week","sku_id","dc_id","actual_demand_units"]),
]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"{name} is missing columns: {missing}. Please upload a correct file.")
        st.stop()

# Normalize time
for df in [inv_hist, shortages, forecasts, actuals]:
    df["ds"] = week_to_date(df["week"])

# Add display names
inv_hist = add_names(inv_hist, master_skus, dcs)
shortages = add_names(shortages, master_skus, dcs)
forecasts = add_names(forecasts, master_skus, dcs)
actuals = add_names(actuals, master_skus, dcs)

# Selections
sku_options = sorted(inv_hist["sku_id"].unique().tolist())
dc_options = sorted(inv_hist["dc_id"].unique().tolist())
with st.sidebar:
    st.subheader("Filters")
    sku_sel = st.multiselect("SKU(s)", sku_options, default=sku_options)
    dc_sel = st.multiselect("DC(s)", dc_options, default=dc_options)
    weeks_all = sorted(inv_hist["week"].unique().tolist())
    wk_min, wk_max = (min(weeks_all), max(weeks_all)) if weeks_all else (1,1)
    wk_range = st.slider("Week Range", min_value=int(wk_min), max_value=int(wk_max), value=(int(wk_min), int(wk_max)), step=1)

    safety_mult = st.slider("Safety Stock Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05, help="Applies on top of baseline safety stock")

# Filtered views
fmask = inv_hist["sku_id"].isin(sku_sel) & inv_hist["dc_id"].isin(dc_sel) & inv_hist["week"].between(wk_range[0], wk_range[1])
inv_f = inv_hist.loc[fmask].copy()
short_f = shortages.loc[fmask].copy()
fc_f = forecasts.loc[fmask].copy()
act_f = actuals.loc[fmask].copy()

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Safety Stock", "Forecasting", "Simulator", "Strategies"])

# ---------------------------------------------------------
# Tab 1: Overview (Inventory & Shortages)
# ---------------------------------------------------------
with tab1:
    st.subheader("Inventory & Shortages (Filtered)")
    col1, col2 = st.columns([2, 1])

    with col1:
        color_col = "sku_name" if "sku_name" in inv_f.columns else "sku_id"
        line_group = "dc_name" if "dc_name" in inv_f.columns else "dc_id"
        fig1 = px.line(inv_f, x="ds", y="inventory_end_units", color=color_col, line_group=line_group,
                       hover_data=["week","sku_id","dc_id","dc_name","sku_name"], markers=True,
                       title="Ending Inventory Over Time")
        fig1.update_layout(xaxis_title="Week", yaxis_title="Units")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(short_f, x="ds", y="shortage_units", color=color_col,
                      hover_data=["week","sku_id","dc_id","dc_name","sku_name"],
                      title="Shortages by Week", barmode="group")
        fig2.update_layout(xaxis_title="Week", yaxis_title="Shortage Units")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        inv_total = inv_f["inventory_end_units"].sum()
        short_total = short_f["shortage_units"].sum()
        service_lvl = 100.0 * (1 - (short_total / (short_total + act_f["actual_demand_units"].sum() + 1e-9)))
        st.metric("Total Inventory (Units)", f"{int(inv_total):,}")
        st.metric("Total Shortages (Units)", f"{int(short_total):,}")
        st.metric("Service Level (%)", f"{service_lvl:.2f}")
        st.download_button("Download Filtered Inventory CSV", inv_f.to_csv(index=False), file_name="filtered_inventory.csv")
        st.download_button("Download Filtered Shortages CSV", short_f.to_csv(index=False), file_name="filtered_shortages.csv")

# ---------------------------------------------------------
# Tab 2: Safety Stock (NEW)
# ---------------------------------------------------------
with tab2:
    st.subheader("Safety Stock Planning")

    if safety_stock.empty or inv_wk1.empty:
        st.warning("Missing `p1_safety_stock.csv` or `p1_inventory_wk1.csv`. Upload them to use this tab.")
    else:
        ss = safety_stock.copy()
        ss = add_names(ss, master_skus, dcs)
        onhand = inv_wk1.copy()
        onhand = add_names(onhand, master_skus, dcs)

        # Merge to show baseline On-Hand and parameters
        ss_merged = ss.merge(onhand[["sku_id","dc_id","on_hand_units_wk1"]], on=["sku_id","dc_id"], how="left")
        ss_merged["adj_safety_stock"] = (ss_merged["safety_stock_units"] * safety_mult).round().astype(int)
        # Reorder point: mean demand during lead time + adj safety stock
        ss_merged["demand_during_LT"] = (ss_merged["mean"] * ss_merged["replen_lead_time_weeks"]).round()
        ss_merged["reorder_point"] = (ss_merged["demand_during_LT"] + ss_merged["adj_safety_stock"]).astype(int)
        ss_merged["weeks_of_coverage"] = (ss_merged["on_hand_units_wk1"] / (ss_merged["mean"] + 1e-9)).round(2)

        # Filters
        c1, c2 = st.columns(2)
        with c1:
            sku_pick = st.selectbox("Select SKU", sorted(ss_merged["sku_id"].unique()), format_func=lambda s: ss_merged.loc[ss_merged["sku_id"]==s,"sku_name"].iloc[0] if "sku_name" in ss_merged.columns else s)
        with c2:
            dc_pick = st.selectbox("Select DC", sorted(ss_merged["dc_id"].unique()), format_func=lambda d: ss_merged.loc[ss_merged["dc_id"]==d,"dc_name"].iloc[0] if "dc_name" in ss_merged.columns else d)

        view = ss_merged[(ss_merged["sku_id"]==sku_pick) & (ss_merged["dc_id"]==dc_pick)]

        st.markdown(f"**Safety Stock Multiplier Applied:** `{safety_mult:.2f}x`")
        st.dataframe(view[["sku_id","sku_name","dc_id","dc_name","on_hand_units_wk1","mean","std","replen_lead_time_weeks","safety_stock_units","adj_safety_stock","demand_during_LT","reorder_point","weeks_of_coverage"]])

        # Visual: ROP vs On-Hand
        fig = go.Figure()
        fig.add_bar(name="On-hand (Wk1)", x=["On-hand"], y=[int(view["on_hand_units_wk1"].iloc[0])])
        fig.add_bar(name="Reorder Point (Adj)", x=["Reorder Point"], y=[int(view["reorder_point"].iloc[0])])
        fig.update_layout(barmode="group", title="On-hand vs Reorder Point", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)

        # Full table download
        st.download_button("Download Safety Stock Table", ss_merged.to_csv(index=False), file_name="safety_stock_planning.csv")

# ---------------------------------------------------------
# Tab 3: Forecasting (Prophet optional)
# ---------------------------------------------------------
with tab3:
    st.subheader("Forecasting (Inventory via Prophet)")
    if not PROPHET_AVAILABLE:
        st.info("`prophet` is not installed in this environment. Upload with it installed to enable this tab.")
    else:
        sku_pick = st.selectbox("SKU for Forecast", sku_options)
        dc_pick = st.selectbox("DC for Forecast", dc_options)
        series = inv_hist[(inv_hist["sku_id"]==sku_pick) & (inv_hist["dc_id"]==dc_pick)].sort_values("ds")
        if len(series) < 6:
            st.warning("Need at least 6 observations to train Prophet.")
        else:
            dfp = series[["ds","inventory_end_units"]].rename(columns={"ds":"ds","inventory_end_units":"y"})
            m = Prophet()
            m.fit(dfp)
            future = m.make_future_dataframe(periods=8, freq="W")
            fc = m.predict(future)
            st.plotly_chart(plot_plotly(m, fc), use_container_width=True)

# ---------------------------------------------------------
# Tab 4: Simulator (uses Safety Multiplier)
# ---------------------------------------------------------
with tab4:
    st.subheader("Simple Demand–Supply Simulator")
    st.caption("Simulates weekly shortages using forecast minus available inventory with adjustable safety stock multiplier.")

    # Build a simple joint frame week × sku × dc
    base_df = forecasts.merge(inv_hist[["sku_id","dc_id","week","inventory_end_units"]], on=["sku_id","dc_id","week"], how="left", suffixes=("","_inv"))
    base_df = base_df.merge(shortages, on=["sku_id","dc_id","week"], how="left", suffixes=("","_short"))
    if not safety_stock.empty:
        ss_use = safety_stock.copy()
        ss_use["adj_safety_stock"] = (ss_use["safety_stock_units"] * safety_mult).round()
        base_df = base_df.merge(ss_use[["sku_id","dc_id","adj_safety_stock"]], on=["sku_id","dc_id"], how="left")
    else:
        base_df["adj_safety_stock"] = 0

    base_df["available"] = (base_df["inventory_end_units"].fillna(0) + base_df["adj_safety_stock"].fillna(0))
    base_df["sim_shortage"] = (base_df["forecast_units"] - base_df["available"]).clip(lower=0)
    heat = base_df.pivot_table(index="dc_name" if "dc_name" in base_df.columns else "dc_id",
                               columns="week", values="sim_shortage", aggfunc="sum").fillna(0)

    met1, met2, met3 = st.columns(3)
    met1.metric("Total Forecast (filtered)", f"{int(fc_f['forecast_units'].sum()):,}")
    met2.metric("Total Simulated Shortage", f"{int(base_df['sim_shortage'].sum()):,}")
    demanded = act_f["actual_demand_units"].sum()
    service = 100.0 * (1 - (base_df["sim_shortage"].sum() / (base_df["forecast_units"].sum() + 1e-9)))
    met3.metric("Simulated Service Level (%)", f"{service:.2f}")

    st.plotly_chart(px.imshow(heat, aspect="auto", title="Simulated Shortages Heatmap (DC × Week)"), use_container_width=True)

    st.download_button("Download Simulation Table", base_df.to_csv(index=False), file_name="sim_output.csv")

# ---------------------------------------------------------
# Tab 5: Strategy Formulation (NEW)
# ---------------------------------------------------------
with tab5:
    st.subheader("Strategy Formulation")
    st.caption("Auto-derived recommendations from data + adjustable safety stock multiplier.")

    # 1) Capacity Planning: compare total weekly forecast vs plant capacity
    cap_rec = []
    if not plant_capacity.empty:
        total_fc = forecasts.groupby("week")["forecast_units"].sum().rename("total_forecast")
        cap_df = plant_capacity.groupby("week")["capacity_units"].sum().rename("total_capacity")
        cap_join = pd.concat([total_fc, cap_df], axis=1).dropna()
        cap_join["gap"] = cap_join["total_forecast"] - cap_join["total_capacity"]
        peak_weeks = cap_join[cap_join["gap"] > 0]
        if not peak_weeks.empty:
            cap_rec.append(f"Capacity shortfall in weeks: {', '.join(map(str, peak_weeks.index.tolist()))}. "
                           f"Avg shortfall: {int(peak_weeks['gap'].mean()):,} units. Consider subcontracting or overtime.")
        else:
            cap_rec.append("Current plant capacity covers weekly forecast. Maintain preventive maintenance and shift balancing.")
    else:
        cap_rec.append("Upload `p1_plant_capacity.csv` to get capacity recommendations.")

    # 2) Inventory Planning: stock-outs vs overstock
    inv_avg = actuals.groupby(["sku_id","dc_id"])["actual_demand_units"].mean().rename("avg_demand")
    inv_last = inv_hist.groupby(["sku_id","dc_id"])["inventory_end_units"].mean().rename("avg_inventory")
    inv_join = pd.concat([inv_avg, inv_last], axis=1).dropna()
    inv_join["inv_to_demand_ratio"] = inv_join["avg_inventory"] / (inv_join["avg_demand"] + 1e-9)
    risky_under = inv_join[inv_join["inv_to_demand_ratio"] < 0.6].reset_index()
    risky_over = inv_join[inv_join["inv_to_demand_ratio"] > 1.8].reset_index()

    # 3) Supplier delays: low OTD or long lead time
    sup_rec = []
    if not suppliers.empty:
        late = suppliers[(suppliers["on_time_delivery_rate"] < 0.92) | (suppliers["supplier_lead_time_weeks"] > 2)]
        if not late.empty:
            for _, r in late.iterrows():
                sup_rec.append(f"SKU {r['sku_id']}: Improve supplier {r['supplier_id']} (OTD={r['on_time_delivery_rate']:.2f}, LT={int(r['supplier_lead_time_weeks'])}w). Consider dual sourcing & buffer.")
        else:
            sup_rec.append("Suppliers meet OTD≥0.92 and LT≤2w. Maintain SRM and quarterly reviews.")
    else:
        sup_rec.append("Upload `p1_supplier_data.csv` to get supplier recommendations.")

    # 4) Demand-Supply matching during peaks
    surge = (actuals.merge(forecasts, on=["week","sku_id","dc_id"], suffixes=("_act","_fc")))
    surge["delta"] = (surge["actual_demand_units"] - surge["forecast_units"]) / (surge["forecast_units"] + 1e-9)
    surge_weeks = surge.groupby("week")["delta"].mean()
    surge_weeks = surge_weeks[surge_weeks > 0.2]  # weeks where actuals exceed forecast by >20% on average
    if len(surge_weeks) > 0:
        surge_msg = f"High demand surge (>20%) seen in weeks: {', '.join(map(lambda x: str(int(x)), surge_weeks.index.tolist()))}. "\
                    "Actions: pre-build inventory, temporary allocations to high-demand DCs, demand shaping (promotions control)."
    else:
        surge_msg = "No major surge weeks (>20%) detected on average. Maintain rolling forecast and S&OP."

    # Cards / bullets
    st.markdown("### Capacity Planning")
    for m in cap_rec:
        st.info("• " + m)

    st.markdown("### Inventory Rebalancing")
    if not risky_under.empty:
        st.warning(f"Stock-out risk (low inventory vs demand): {len(risky_under)} SKU–DC pairs. Example top 5:")
        st.dataframe(add_names(risky_under, master_skus, dcs).head(5))
    else:
        st.success("No SKU–DC pair has inventory < 60% of average demand.")

    if not risky_over.empty:
        st.warning(f"Overstock risk (high inventory vs demand): {len(risky_over)} SKU–DC pairs. Example top 5:")
        st.dataframe(add_names(risky_over, master_skus, dcs).head(5))
    else:
        st.success("No SKU–DC pair has inventory > 1.8× of average demand.")

    st.markdown("### Supplier Management")
    for m in sup_rec:
        st.info("• " + m)

    st.markdown("### Demand–Supply Matching")
    st.info("• " + surge_msg)

    st.markdown("### Technology & Collaboration")
    st.caption("Adopt integrated planning: ERP + AI Forecasting + APS. Cross-functional weekly S&OP, DC–Plant visibility, supplier portals, and alerting on service KPIs.")

st.caption("© FreshBites demo — improved with Safety Stock + Strategy tabs. Upload your own CSVs to override the demo data.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.express as px

from main import build_dashboard_state

st.set_page_config(page_title="Sunrise Inventory Control", layout="wide")

st.title("📊 Sunrise Distributors — Inventory Control Center")


@st.cache_data(show_spinner="Building dashboard data...")
def load_dashboard_state():
    return build_dashboard_state()


state = load_dashboard_state()
sku_master = state['sku_master']
inventory = state['inventory']
all_forecasts = state['all_forecasts']
reorder_df = state['reorder_df']
diwali_retrospective = state['diwali_retrospective']

# === SIDEBAR: SKU Selector ===
sku_selected = st.sidebar.selectbox(
    "Select SKU",
    sku_master.sku_id.unique(),
    format_func=lambda x: f"{x} — {sku_master[sku_master.sku_id == x].product_name.iloc[0]}"
)

# === MAIN AREA: Forecast Chart ===
forecast_data = all_forecasts[sku_selected].copy()
forecast_data['Week'] = forecast_data.ds.dt.strftime('%b %d')

fig = px.line(
    forecast_data,
    x='Week',
    y='yhat',
    title=f'6-Week Forecast: {sku_selected}',
    labels={'yhat': 'Predicted Sales (units)'}
)
fig.add_scatter(x=forecast_data.Week, y=forecast_data.yhat_upper, name='Upper Bound', line=dict(dash='dash'))
fig.add_scatter(x=forecast_data.Week, y=forecast_data.yhat_lower, name='Lower Bound', line=dict(dash='dash'))
st.plotly_chart(fig, use_container_width=True)

# === STOCK VS FORECAST ===
col1, col2, col3 = st.columns(3)

current_stock = inventory[inventory.sku_id == sku_selected].warehouse_stock.iloc[0]
forecast_6wk_total = forecast_data.yhat.sum()
reorder_qty = reorder_df[reorder_df.sku_id == sku_selected].order_qty.iloc[0]

col1.metric("Current Stock", f"{current_stock} units")
col2.metric("6-Week Forecast", f"{forecast_6wk_total:.0f} units", delta=f"{forecast_6wk_total - current_stock:.0f}")
col3.metric("Recommended Order", f"{reorder_qty} units")

# === DIWALI RETROSPECTIVE HEATMAP ===
st.subheader("🎆 Diwali 2023 Stockout Analysis (40% of Score)")

heatmap_data = diwali_retrospective.copy()
fig2 = px.bar(
    heatmap_data,
    x='sku_id',
    y='stockout_score',
    color='stockout_score',
    title='Top 14 Predicted Stockouts (Signature Detection)',
    labels={'stockout_score': 'Stockout Probability'}
)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(heatmap_data[['sku_id', 'pre_velocity', 'diwali_sales', 'post_velocity', 'stockout_score']])
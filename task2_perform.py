import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import json

# ─── Helper Functions ────────────────────────────────────────────────
def load_csv(file):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file)

def preprocess(df):
    """Preprocess the DataFrame."""
    df = df.copy()
    # Rename columns for consistency
    rename_map = {
        'Location name': 'Store',
        'Task name': 'Task',
        'Task status_x': 'Status',
        'End date': 'Due Date',
        'Date completed': 'Completed Date'
    }
    df = df.rename(columns=rename_map)
    
    # Convert date columns to datetime
    df['Due Date'] = pd.to_datetime(df['Due Date'], errors='coerce')
    df['Completed Date'] = pd.to_datetime(df['Completed Date'], errors='coerce')
    
    # Calculate Days Before Due
    current_date = pd.Timestamp.now()
    df['Days Before Due'] = (df['Due Date'] - current_date).dt.total_seconds() / (24 * 3600)
    df.loc[df['Status'] == 'Completed', 'Days Before Due'] = (
        (df['Due Date'] - df['Completed Date']).dt.total_seconds() / (24 * 3600)
    )
    
    return df

def metric_card(label, value, delta=None):
    """Display a metric card with optional delta."""
    delta_str = f"{delta:+.0%}" if delta is not None else None
    st.metric(label, value, delta=delta_str)

# ─── Sidebar / Upload ────────────────────────────────────────────────
st.sidebar.header("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file = st.sidebar.file_uploader("📊 KPI CSV (optional)", type="csv")

if not task_file:
    st.sidebar.info("Please upload your Task CSV to get started")
    st.stop()

# Load and preprocess data
df = load_csv(task_file)
df = preprocess(df)

# Load KPI data if provided
kpi_df = load_csv(kpi_file) if kpi_file else None

# ─── Main Layout ────────────────────────────────────────────────────
st.title("Retail Task Dashboard")
tab1, tab2, tab3 = st.tabs(["Overview", "Store Health", "Trends & Forecast"])

# ─── Tab 1: Overview ────────────────────────────────────────────────
with tab1:
    st.header("Task Overview")
    
    # Task Status Distribution
    status_counts = df.groupby(['Store', 'Status']).size().unstack(fill_value=0)
    fig_status = px.bar(
        status_counts,
        x=status_counts.index,
        y=status_counts.columns,
        title="Task Status by Store",
        labels={"value": "Number of Tasks", "Store": "Store"},
        template="plotly_white"
    )
    st.plotly_chart(fig_status, use_container_width=True)
    
    # Task Category Breakdown
    category_counts = df.groupby(['Store', 'Task category']).size().unstack(fill_value=0)
    fig_category = px.bar(
        category_counts,
        x=category_counts.index,
        y=category_counts.columns,
        title="Task Categories by Store",
        labels={"value": "Number of Tasks", "Store": "Store"},
        template="plotly_white"
    )
    st.plotly_chart(fig_category, use_container_width=True)

# ─── Tab 2: Store Health ────────────────────────────────────────────
with tab2:
    st.header("Store Health")
    
    # Calculate on-time completion rate
    health_data = (
        df.groupby('Store')
          .agg(OnTimeRate=('Days Before Due', lambda x: (x >= 0).mean() if not x.empty else 0))
          .reset_index()
    )
    
    # Merge with KPI data if available
    if kpi_df is not None:
        kpi_health = kpi_df.groupby('Store').agg({
            'Sales vs Target': 'mean',
            'CSAT': 'mean'
        }).reset_index()
        health_data = health_data.merge(kpi_health, on='Store', how='left')
        health_data['Health Score'] = (
            health_data['OnTimeRate'] * 0.4 +
            health_data['Sales vs Target'].fillna(0) * 0.3 +
            health_data['CSAT'].fillna(0) * 0.3
        )
    else:
        health_data['Health Score'] = health_data['OnTimeRate']
    
    # Health Score Visualization
    fig_health = px.bar(
        health_data,
        x='Store',
        y='Health Score',
        title="Store Health Scores",
        labels={"Health Score": "Health Score"},
        template="plotly_white",
        color='Health Score',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_health, use_container_width=True)
    
    # Hierarchy Visualization
    hierarchy_cols = [col for col in df.columns if col.startswith('Level')]
    if hierarchy_cols:
        hierarchy_data = df[hierarchy_cols + ['Store', 'Health Score']].drop_duplicates()
        fig_hierarchy = px.treemap(
            hierarchy_data,
            path=hierarchy_cols + ['Store'],
            values='Health Score',
            title="Health Score by Hierarchy",
            color='Health Score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_hierarchy, use_container_width=True)

# ─── Tab 3: Trends & Forecast ────────────────────────────────────────
with tab3:
    st.header("Trends & Predictive Insights")
    
    # Weekly On-Time Rate Trend
    st.subheader("Historical On-Time Rate")
    df['Week Start'] = pd.to_datetime(df['Due Date']).dt.to_period('W').apply(lambda r: r.start_time)
    trend = (
        df.groupby('Week Start')
          .apply(lambda d: d.groupby('Task ID')['Days Before Due'].max().ge(0).mean() if not d.empty else 0)
          .rename("OnTimeRate")
          .reset_index()
          .sort_values('Week Start')
    )
    fig_trend = px.line(
        trend,
        x='Week Start',
        y='OnTimeRate',
        title="Weekly On-Time Rate Trend",
        labels={"OnTimeRate": "On-Time Rate (%)"},
        template="plotly_white"
    )
    fig_trend.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    if trend.empty:
        st.warning("No completed tasks to establish a trend. Predictions will assume current overdue status.")
    
    # Forecast On-Time Rate for Next Week
    st.subheader("Forecasted On-Time Rate (Next Week)")
    x = np.arange(len(trend)).reshape(-1, 1)
    y = trend['OnTimeRate']
    if len(trend) > 1:
        model = LinearRegression().fit(x, y)
        pred = model.predict([[len(trend)]])[0]
        pred = max(0, min(1, pred))
        prev_rate = y.iloc[-1] if not y.empty else 0
        delta = pred - prev_rate
        metric_card("Forecasted On-Time Rate", f"{pred:.0%}", delta)
    else:
        current_overdue = (pd.to_datetime(df['Due Date']) < pd.Timestamp.now()).mean()
        pred = 1 - current_overdue
        metric_card("Forecasted On-Time Rate", f"{pred:.0%}", None)
    
    # Store-Specific Completion Rate Forecast
    st.subheader("Store-Specific Completion Rate Forecast")
    store_trends = (
        df.groupby(['Week Start', 'Store'])
          .apply(lambda d: d.groupby('Task ID')['Days Before Due'].max().ge(0).mean() if not d.empty else 0)
          .rename("OnTimeRate")
          .reset_index()
    )
    forecast_data = []
    for store in store_trends['Store'].unique():
        store_data = store_trends[store_trends['Store'] == store]
        if len(store_data) > 1:
            x_store = np.arange(len(store_data)).reshape(-1, 1)
            y_store = store_data['OnTimeRate']
            model_store = LinearRegression().fit(x_store, y_store)
            pred_store = model_store.predict([[len(store_data)]])[0]
            forecast_data.append({
                "Store": store,
                "Forecasted On-Time Rate": max(0, min(1, pred_store))
            })
        else:
            store_tasks = df[df['Store'] == store]
            current_overdue = (pd.to_datetime(store_tasks['Due Date']) < pd.Timestamp.now()).mean()
            pred_store = 1 - current_overdue
            forecast_data.append({"Store": store, "Forecasted On-Time Rate": pred_store})
    
    forecast_df = pd.DataFrame(forecast_data)
    fig_store = px.bar(
        forecast_df,
        x='Store',
        y='Forecasted On-Time Rate',
        title="Predicted On-Time Rate by Store (Next Week)",
        labels={"Forecasted On-Time Rate": "On-Time Rate (%)"},
        template="plotly_white"
    )
    fig_store.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_store, use_container_width=True)
    
    # Health Score Forecast
    st.subheader("Forecasted Store Health Scores")
    health_data = (
        df.groupby(['Week Start', 'Store'])
          .agg(OnTimeRate=('Days Before Due', lambda x: (x >= 0).mean() if not x.empty else 0))
          .reset_index()
    )
    health_forecast = []
    for store in health_data['Store'].unique():
        store_health = health_data[health_data['Store'] == store]
        if len(store_health) > 1:
            x_health = np.arange(len(store

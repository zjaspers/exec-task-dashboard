import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

# ─── Mobile-Optimized Styling ────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding: 0.5rem 1rem; font-family: 'Inter', sans-serif; }
  .metric-card {
    background: white; border-radius: 8px; padding: 0.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;
    margin: 0.4rem 0;
  }
  .metric-value { font-size: 1.4rem; font-weight: bold; }
  .metric-label { font-size: 0.8rem; color: #555; }
  /* make tabs scrollable on mobile */
  div[data-testid="stHorizontalBlock"] > div:first-child {
    overflow-x: auto; white-space: nowrap;
  }
  div[data-testid="stHorizontalBlock"] > div:first-child > button {
    flex: 0 0 auto; min-width: 100px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def preprocess(df):
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing, 'Days Before Due'] = -((df.loc[missing, 'End date'] - today).dt.days)
    df['Overdue'] = df['Days Before Due'] < 0
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df = df[~df['Store'].isin(['JameTrade','Midwest'])]
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

def metric_card(label, value):
    st.markdown(f"""
      <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
      </div>
    """, unsafe_allow_html=True)

def top_overdue_tasks(df, threshold_days, n=5):
    tasks = (df.groupby('Task ID')
               .agg(Store=('Store','first'),
                    Task=('Task name','first'),
                    DaysBeforeDue=('Days Before Due','max'))
               .reset_index())
    tasks['Days Late'] = -tasks['DaysBeforeDue']
    late = tasks[tasks['Days Late'] >= threshold_days]
    return late.sort_values('Days Late', ascending=False).head(n)[['Store','Task ID','Task','Days Late']]

def flag_stores(df, threshold_days, count_threshold):
    steps = (df.groupby(['Store','Task ID'])
               .agg(DaysBeforeDue=('Days Before Due','max'))
               .reset_index())
    steps['Days Late'] = -steps['DaysBeforeDue']
    late = steps[steps['Days Late'] >= threshold_days]
    counts = (late.groupby('Store')['Task ID']
                  .nunique()
                  .reset_index(name='Overdue Tasks'))
    return counts[counts['Overdue Tasks'] >= count_threshold]

# ─── Sidebar: Data Upload & Dynamic Controls ─────────────────────────
st.sidebar.header("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

overdue_threshold = st.sidebar.number_input(
    "Min days late to flag", min_value=1, max_value=30, value=1)
store_overdue_count = st.sidebar.number_input(
    "Stores with ≥ this many overdue tasks", min_value=1, max_value=20, value=3)

if not task_file:
    st.sidebar.info("Upload Task CSV to begin.")
    st.stop()

# ─── Load & Prepare Data ─────────────────────────────────────────────
df = load_csv(task_file)
df = preprocess(df)
if kpi_file:
    kpi = load_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
    df = df.merge(kpi, on=['Location external ID','Store'], how='left')

# ─── Week Selector & Filters ─────────────────────────────────────────
weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()}–{(w + timedelta(days=6)).date()}" for w in weeks]
sel = st.sidebar.selectbox("Select Week", labels)
start = weeks[labels.index(sel)]
week_df = df[df['Week Start'] == start]

task_list = sorted(week_df['Task name'].unique())
store_list = sorted(week_df['Store'].unique())
sel_tasks = st.sidebar.multiselect("Filter by Task", task_list, default=task_list)
sel_stores = st.sidebar.multiselect("Filter by Store", store_list)

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Metrics & Recos",
    "Store Performance",
    "Task Analysis"
])

# ─── Tab 1: Key Metrics & Dynamic Recommendations ───────────────────
with tab1:
    st.header("Key Metrics")
    total_tasks = filtered['Task ID'].nunique()
    on_time = filtered.groupby('Task ID')['Days Before Due'].max().ge(0).sum()
    avg_days = filtered.groupby('Task ID')['Days Before Due'].mean().mean().round(1)
    overdue_count = total_tasks - on_time
    adhoc = filtered.groupby('Task ID')['Store'].nunique().eq(1).sum()
    avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None

    cols = st.columns(6)
    with cols[0]: metric_card("Total Tasks", total_tasks)
    with cols[1]: metric_card("% On Time", f"{on_time/total_tasks:.0%}")
    with cols[2]: metric_card("Avg Days Early/Late", avg_days)
    with cols[3]: metric_card("Overdue Tasks", overdue_count)
    with cols[4]: metric_card("Ad Hoc Tasks", adhoc)
    if avg_csat is not None:
        with cols[5]: metric_card("Avg CSAT", f"{avg_csat:.1f}")

    st.markdown("### Underperforming Tasks")
    underperf = top_overdue_tasks(filtered, overdue_threshold, n=5)
    st.table(underperf)

    st.markdown("### Underperforming Stores")
    bad_stores = flag_stores(filtered, overdue_threshold, store_overdue_count)
    st.table(bad_stores)

    st.markdown("### Actionable Recommendations")
    if not underperf.empty:
        st.markdown(f"- **Standardize top {len(underperf)} break-fix tasks**: Create templates for these tasks.")
    if not bad_stores.empty:
        stores = ", ".join(bad_stores['Store'])
        st.markdown(f"- **Initiate manager check-ins for:** {stores}.")
    if underperf.empty and bad_stores.empty:
        st.markdown("- 🎉 All tasks and stores are within thresholds!")

# ─── Tab 2: Store Performance ────────────────────────────────────────
with tab2:
    st.header("Store Performance")
    sb = (filtered.groupby('Store')
              .agg(Total_Tasks=('Task ID','nunique'),
                   Overdue_Rate=('Overdue','mean'))
              .reset_index()
              .sort_values('Overdue_Rate', ascending=False))
    st.bar_chart(sb.set_index('Store')['Overdue_Rate'])
    sb['Overdue_Rate'] = sb['Overdue_Rate'].map("{:.0%}".format)
    st.table(sb)

# ─── Tab 3: Task Analysis ────────────────────────────────────────────
with tab3:
    st.header("Task Effort & Performance")
    ta = (filtered.groupby('Task name')
              .agg(Count=('Task ID','nunique'),
                   Effort=('Expected duration','sum'),
                   Overdue_Rate=('Overdue','mean'))
              .reset_index()
              .sort_values('Effort', ascending=False))
    st.bar_chart(ta.set_index('Task name')['Effort'])
    ta['Overdue_Rate'] = ta['Overdue_Rate'].map("{:.0%}".format)
    st.table(ta)

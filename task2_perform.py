import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta, date
from sklearn.linear_model import LinearRegression

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

# ─── Helpers ─────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def preprocess(df):
    df['End date']       = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing,'Days Before Due'] = -((df.loc[missing,'End date'] - today).dt.days)
    df['Overdue'] = df['Days Before Due'] < 0
    df['Region']  = df['Level 1'].fillna('Unknown')
    df['Store']   = df['Location name']
    df = df[~df['Store'].isin(['JameTrade','Midwest'])]
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

def metric_card(label, value, delta=None):
    html = f"<div class='metric-card'><div class='metric-value'>{value}"
    if delta is not None:
        arrow = "▲" if delta>0 else "▼"
        color = "#2ca02c" if delta>0 else "#d62728"
        html += f" <span style='color:{color};font-size:0.8rem'>{arrow}{abs(delta):.0%}</span>"
    html += f"</div><div class='metric-label'>{label}</div></div>"
    st.markdown(html, unsafe_allow_html=True)

# ─── Sidebar: Data Upload & Controls ─────────────────────────────────
st.sidebar.header("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

if not task_file:
    st.sidebar.info("Upload Task CSV to begin.")
    st.stop()

# ─── Load & Prepare ──────────────────────────────────────────────────
df = load_csv(task_file)
df = preprocess(df)
if kpi_file:
    kpi = load_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
    df = df.merge(kpi, on=['Location external ID','Store'], how='left')

# Week selector
weeks  = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()}–{(w+timedelta(days=6)).date()}" for w in weeks]
sel    = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(sel)]
week_df = df[df['Week Start']==start]

# Task/store filters
tasks  = sorted(week_df['Task name'].unique())
stores = sorted(week_df['Store'].unique())
sel_tasks  = st.sidebar.multiselect("Filter by Task", tasks, default=tasks)
sel_stores = st.sidebar.multiselect("Filter by Store", stores)

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Key Metrics & Recos",
    "🏥 Store Health",
    "🔮 Trends & Forecast",
    "✅ Action Tracker"
])

# ─── Tab 1: Key Metrics & Recommendations ───────────────────────────
with tab1:
    st.header("Key Metrics")
    total = filtered['Task ID'].nunique()
    on_time = filtered.groupby('Task ID')['Days Before Due'].max().ge(0).sum()
    avg_days = filtered.groupby('Task ID')['Days Before Due'].mean().mean().round(1)
    overdue = total - on_time
    adhoc = filtered.groupby('Task ID')['Store'].nunique().eq(1).sum()
    avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None

    # week-over-week on-time delta
    prev = df[df['Week Start']==start - timedelta(weeks=1)]
    prev_on = prev.groupby('Task ID')['Days Before Due'].max().ge(0).sum() if not prev.empty else None
    delta = None if prev_on is None else (on_time - prev_on) / prev_on

    cols = st.columns(6)
    with cols[0]: metric_card("Total Tasks", total, None)
    with cols[1]: metric_card("% On Time", f"{on_time/total:.0%}", delta)
    with cols[2]: metric_card("Avg Days Before Due", f"{avg_days:.1f}")
    with cols[3]: metric_card("Overdue Tasks", overdue, None)
    with cols[4]: metric_card("Ad Hoc Tasks", adhoc, None)
    if avg_csat is not None:
        with cols[5]: metric_card("Avg CSAT", f"{avg_csat:.1f}")

    st.markdown("### Top 5 Overdue Tasks")
    over = (filtered.groupby('Task ID')
            .agg(Store=('Store','first'),
                 Task=('Task name','first'),
                 DaysLate=('Days Before Due','max'))
            .reset_index())
    over = over[over['DaysLate']<0]
    over['Days Late'] = -over['DaysLate']
    st.table(over[['Store','Task ID','Task','Days Late']].sort_values('Days Late', ascending=False).head(5))

    st.markdown("### Actionable Recommendations")
    st.markdown("- Standardize break-fix tasks: build templates for the 5 tasks above.")
    st.markdown("- Schedule weekly check-ins with store leads for overdue items.")
    st.markdown("- Enable region managers to monitor their stores’ task boards.")

# ─── Tab 2: Store Health Overview ────────────────────────────────────
with tab2:
    st.header("Store Health Overview")
    # Composite health score
    sb = filtered.groupby('Store').agg(
        OnTimeRate=('Days Before Due', lambda x: (x>=0).mean()),
        CSAT=('CSAT Score', 'mean'),
        Sales=('Sales vs Target (%)', 'mean'),
        TaskLoad=('Task ID', 'nunique')
    ).reset_index().fillna(0)
    # normalize and combine
    sb['HealthScore'] = (
        sb['OnTimeRate']*0.4 +
        (sb['CSAT']/100)*0.3 +
        (sb['Sales']/sb['Sales'].max())*0.2 +
        (1 - sb['TaskLoad']/sb['TaskLoad'].max())*0.1
    )
    # Top & bottom 5
    top5 = sb.nlargest(5,'HealthScore')[['Store','HealthScore']]
    bot5 = sb.nsmallest(5,'HealthScore')[['Store','HealthScore']]

    st.subheader("Top 5 Stores")
    st.table(top5.style.format({'HealthScore':'{:.2f}'}))
    st.subheader("Bottom 5 Stores")
    st.table(bot5.style.format({'HealthScore':'{:.2f}'}))

    st.subheader("Region Health Matrix")
    reg = sb.groupby('Store')['HealthScore'].mean().reset_index()  # if region in data, group by region
    # placeholder: use store names vs score as heatmap rows
    plt.figure(figsize=(4, max(2,len(reg)*0.2)))
    sns.heatmap(reg.set_index('Store'), annot=True, cmap='RdYlGn_r', cbar=False, linewidths=0.5)
    st.pyplot(plt.gcf())

# ─── Tab 3: Trends & Forecast ────────────────────────────────────────
with tab3:
    st.header("Weekly Compliance Trend")
    trend = (df.groupby('Week Start')
               .apply(lambda d: d.groupby('Task ID')['Days Before Due'].max().ge(0).mean())
               .rename("OnTimeRate")
               .reset_index())
    trend = trend.sort_values('Week Start')
    st.line_chart(trend.set_index('Week Start')['OnTimeRate'])

    st.subheader("Forecasted Overdue Tasks Next Week")
    # build forecast model
    x = np.arange(len(trend)).reshape(-1,1)
    y = trend['OnTimeRate']
    if len(trend)>1:
        model = LinearRegression().fit(x, y)
        pred = model.predict([[len(trend)]])[0]
        st.metric("Forecasted On-Time Rate", f"{pred:.0%}")
    # simple bar of upcoming effort needed
    fut = (filtered.groupby('Store')['Expected duration'].sum() *
           (1-pred)).reset_index(name='Required Effort (hrs)')
    st.bar_chart(fut.set_index('Store')['Required Effort (hrs)'])

# ─── Tab 4: Action & Accountability Tracker ──────────────────────────
with tab4:
    st.header("Action & Accountability Tracker")
    # For each overdue task, suggest next action date
    actions = over[['Store','Task ID','Task','Days Late']].copy()
    actions['Next Action Date'] = pd.Timestamp(date.today() + timedelta(days=2))
    actions['Owner'] = "Store Manager"
    st.table(actions)

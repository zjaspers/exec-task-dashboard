import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.linear_model import LinearRegression

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

# ─── Styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding: 1rem; font-family: 'Inter', sans-serif; }
  .metric-card {
    display: inline-block; width: 150px; background: #fff; border-radius: 8px;
    padding: 0.75rem; margin: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .metric-value { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.25rem; }
  .metric-label { font-size: 0.85rem; color: #555; }
  .rec-card {
    background: #f9f9f9; border-left: 4px solid; padding: 0.75rem;
    margin: 0.5rem 0; border-radius: 4px;
  }
  .rec-high { border-color: #d62728; }
  .rec-med  { border-color: #ff7f0e; }
  .rec-low  { border-color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv(f): return pd.read_csv(f)

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

def metric_card(label, value, delta=None, spark=None):
    html = f"<div class='metric-card'><div class='metric-value'>{value}"
    if delta is not None:
        arrow = "▲" if delta>0 else "▼"
        color = "#2ca02c" if delta>0 else "#d62728"
        html += f" <span style='color:{color};font-size:0.8rem'>{arrow}{abs(delta):.0%}</span>"
    html += f"</div><div class='metric-label'>{label}</div></div>"
    st.markdown(html, unsafe_allow_html=True)
    if spark is not None:
        st.line_chart(spark, height=40, use_container_width=True)

# ─── Sidebar: Upload & Filters ───────────────────────────────────────
st.sidebar.header("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

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
weeks  = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()}–{(w+timedelta(days=6)).date()}" for w in weeks]
sel    = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(sel)]
week_df = df[df['Week Start']==start]

tasks  = sorted(week_df['Task name'].unique())
stores = sorted(week_df['Store'].unique())
sel_tasks  = st.sidebar.multiselect("Filter by Task", tasks, default=tasks)
sel_stores = st.sidebar.multiselect("Filter by Store", stores)

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# ─── Compute Week-over-Week for Metrics ─────────────────────────────
prev = df[df['Week Start']==(start - timedelta(weeks=1))]
def week_metric(fn):
    curr = fn(filtered)
    old  = fn(prev) if not prev.empty else None
    if old is not None:
        return curr, (curr - old)/old
    return curr, None

total, d_total   = week_metric(lambda d: d['Task ID'].nunique())
on_time, d_on    = week_metric(lambda d: d.groupby('Task ID')['Days Before Due'].max().ge(0).sum())
avg_days, _       = week_metric(lambda d: d.groupby('Task ID')['Days Before Due'].mean().mean())
overdue, d_over   = week_metric(lambda d: total - on_time)
adhoc, _          = week_metric(lambda d: d.groupby('Task ID')['Store'].nunique().eq(1).sum())
avg_csat         = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Key Metrics & Recommendations",
    "🏬 Store Performance",
    "🛠 Task Analysis"
])

# ─── Tab 1: Key Metrics & Recommendations ───────────────────────────
with tab1:
    st.header("Key Metrics")
    # Sparkline: total tasks last 6 weeks
    spark = (df.groupby('Week Start')['Task ID']
               .nunique().sort_index().tail(6))
    cols = st.columns(6)
    with cols[0]: metric_card("Total Tasks", total, d_total, spark)
    with cols[1]: metric_card("% On Time", f"{on_time/total:.0%}", d_on)
    with cols[2]: metric_card("Avg Days Before Due", f"{avg_days:.1f}")
    with cols[3]: metric_card("Overdue Tasks", overdue, d_over)
    with cols[4]: metric_card("Ad Hoc Tasks", adhoc)
    if avg_csat is not None:
        with cols[5]: metric_card("Avg CSAT", f"{avg_csat:.1f}")

    st.markdown("### Recommendations")
    recs = []
    if 'CSAT Score' in filtered.columns:
        corr = filtered['Days Before Due'].corr(filtered['CSAT Score'])
        recs.append(("Correlation Speed vs CSAT", f"r = {corr:.2f}", "low"))
    late = filtered.groupby('Store')['Days Before Due'].mean().lt(0)
    if late.any():
        names = late[late].index.tolist()
        recs.append(("Underperformers", f"{', '.join(names)}", "high"))
    if 'Expected duration' in filtered and 'CSAT Score' in filtered:
        eff = filtered.groupby('Store')['Expected duration'].sum()
        cs  = filtered.groupby('Store')['CSAT Score'].mean()
        high = eff[eff>eff.quantile(0.8)].index.tolist()
        recs.append(("Effort vs CSAT", f"High effort: {', '.join(high)}", "medium"))

    for title, text, prio in recs:
        cls = f"rec-card rec-{prio}"
        st.markdown(f"<div class='{cls}'><strong>{title}</strong><br>{text}</div>", unsafe_allow_html=True)

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
    st.dataframe(sb, use_container_width=True)

# ─── Tab 3: Task Analysis ────────────────────────────────────────────
with tab3:
    st.header("Task Analysis Quadrant")
    ta = (filtered.groupby('Task name')
                   .agg(Count=('Task ID','nunique'),
                        Effort=('Expected duration','sum'),
                        Overdue=('Overdue','mean'))
                   .reset_index())
    # Quadrant chart
    fig, ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(data=ta, x='Effort', y='Overdue', size='Count', legend=False, ax=ax)
    # Trendline
    X = ta[['Effort']]; y = ta['Overdue']
    m, b = np.polyfit(ta['Effort'], ta['Overdue'], 1)
    ax.plot(ta['Effort'], m*ta['Effort']+b, color='gray', linestyle='--')
    ax.set_xlabel("Effort (hrs)")
    ax.set_ylabel("Overdue Rate")
    st.pyplot(fig)
    ta['Overdue'] = ta['Overdue'].map("{:.0%}".format)
    st.dataframe(ta, use_container_width=True)

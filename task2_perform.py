import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, date
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ─── Page Config & Mobile CSS ────────────────────────────────────────
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")
st.markdown("""
<style>
  .block-container { padding:0.5rem 1rem; font-family:'Inter',sans-serif }
  .metric-card { background:#fff; border-radius:8px; padding:0.6rem;
                 box-shadow:0 1px 3px rgba(0,0,0,0.1); text-align:center;
                 margin:0.4rem 0; }
  .metric-value { font-size:1.4rem; font-weight:bold; }
  .metric-label { font-size:0.8rem; color:#555; }
  /* scrollable tabs */
  div[data-testid="stHorizontalBlock"] > div:first-child {
    overflow-x:auto; white-space:nowrap;
  }
  div[data-testid="stHorizontalBlock"] > div:first-child > button {
    flex:0 0 auto; min-width:100px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def find_and_rename(df, *keywords, new_name):
    """If any column contains all keywords, rename it to new_name."""
    for c in df.columns:
        lc = c.lower()
        if all(kw.lower() in lc for kw in keywords):
            df.rename(columns={c: new_name}, inplace=True)
            return True
    return False

def preprocess(df):
    # 1) unify date columns
    find_and_rename(df, 'end','date',            new_name='End date')
    find_and_rename(df, 'date','completed',      new_name='Date completed')
    find_and_rename(df, 'status',                new_name='Task status')

    # 2) parse dates
    if 'End date'       in df: df['End date']       = pd.to_datetime(df['End date'], errors='coerce')
    if 'Date completed' in df: df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')

    # 3) compute Days Before Due
    today = pd.Timestamp.now().normalize()
    if 'End date' in df and 'Date completed' in df:
        df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
        missing = df['Date completed'].isna() & df['End date'].notna()
        df.loc[missing, 'Days Before Due'] = -((df.loc[missing,'End date'] - today).dt.days)
    else:
        df['Days Before Due'] = np.nan

    # 4) overdue flag
    if 'End date' in df and 'Task status' in df:
        df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'].str.lower()!='completed')
    else:
        df['Overdue'] = False

    # 5) store & region defaults
    df['Region'] = df.get('Level 1', pd.Series()).fillna('Unknown')
    df['Store']  = df.get('Location name', pd.Series()).fillna('Unknown')
    df = df[~df['Store'].isin(['JameTrade','Midwest'])]

    # 6) Week Start
    if 'End date' in df:
        df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r:r.start_time)
    else:
        df['Week Start'] = pd.NaT

    return df

def metric_card(label, value, delta=None):
    arrow = f" {'▲' if delta>0 else '▼'}{abs(delta):.0%}" if delta is not None else ""
    color = "#2ca02c" if (delta or 0)>0 else "#d62728"
    st.markdown(f"""
      <div class='metric-card'>
        <div class='metric-value'>{value}
          <span style='color:{color};font-size:0.8rem'>{arrow}</span>
        </div>
        <div class='metric-label'>{label}</div>
      </div>
    """, unsafe_allow_html=True)

# ─── Sidebar / Upload ────────────────────────────────────────────────
st.sidebar.header("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 KPI CSV (optional)", type="csv")
if not task_file:
    st.sidebar.info("Upload your Task CSV to get started")
    st.stop()

# ─── Load & Prep ─────────────────────────────────────────────────────
df = load_csv(task_file)
df = preprocess(df)
if kpi_file:
    kpi = load_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
    df = df.merge(kpi, on=['Location external ID','Store'], how='left')

# ─── Week & Filters ──────────────────────────────────────────────────
weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
if not weeks:
    st.error("No valid date data found. Check your headers.")
    st.stop()
labels = [f"{w.date()}–{(w+timedelta(days=6)).date()}" for w in weeks]
sel    = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(sel)]
wk     = df[df['Week Start']==start]

tasks  = sorted(wk['Task name'].unique())
stores = sorted(wk['Store'].unique())
sel_t  = st.sidebar.multiselect("Filter Tasks",  tasks, default=tasks)
sel_s  = st.sidebar.multiselect("Filter Stores", stores)
filtered = wk[wk['Task name'].isin(sel_t)]
if sel_s:
    filtered = filtered[filtered['Store'].isin(sel_s)]

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
  "📊 Key Metrics & Recos",
  "🏥 Store Health",
  "🔮 Trends & Forecast",
  "✅ Action Tracker"
])

# ─── Tab 1 ────────────────────────────────────────────────────────────
with tab1:
    st.header("Key Metrics")
    total    = filtered['Task ID'].nunique()
    on_time  = filtered.groupby('Task ID')['Days Before Due'].max().ge(0).sum()
    avg_days = filtered.groupby('Task ID')['Days Before Due'].mean().mean().round(1)
    overdue  = total - on_time
    adhoc    = filtered.groupby('Task ID')['Store'].nunique().eq(1).sum()
    avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered else None

    prev     = df[df['Week Start']==start - timedelta(weeks=1)]
    prev_on  = prev.groupby('Task ID')['Days Before Due'].max().ge(0).sum() if not prev.empty else None
    delta    = None if prev_on is None else (on_time - prev_on)/prev_on

    cols = st.columns(6)
    with cols[0]: metric_card("Total Tasks", total)
    with cols[1]: metric_card("% On Time", f"{on_time/total:.0%}", delta)
    with cols[2]: metric_card("Avg Days +/- Due", f"{avg_days:.1f}")
    with cols[3]: metric_card("Overdue Tasks", overdue)
    with cols[4]: metric_card("Ad Hoc Tasks", adhoc)
    if avg_csat is not None:
        with cols[5]: metric_card("Avg CSAT", f"{avg_csat:.1f}")

    st.markdown("### Top 5 Overdue Tasks")
    over = (filtered.groupby('Task ID')
              .agg(Store=('Store','first'),
                   Task =('Task name','first'),
                   DaysLate=('Days Before Due','max'))
              .reset_index())
    over = over[over['DaysLate']<0]
    over['Days Late'] = -over['DaysLate']
    st.table(over[['Store','Task ID','Task','Days Late']]
             .sort_values('Days Late', ascending=False).head(5))

    st.markdown("### Actionable Recommendations")
    st.markdown("- Standardize break-fix tasks: build templates for those above.")
    st.markdown("- Schedule weekly check-ins with store leads for overdue items.")
    st.markdown("- Enable region managers to monitor their stores’ task boards.")

# ─── Tab 2 ────────────────────────────────────────────────────────────
with tab2:
    st.header("Store Health Overview")
    sb = (filtered.groupby('Store')
          .agg(OnTimeRate=('Days Before Due', lambda x:(x>=0).mean()),
               CSAT      =('CSAT Score','mean'),
               Sales     =('Sales vs Target (%)','mean'),
               TaskLoad  =('Task ID','nunique'))
          .reset_index().fillna(0))
    sb['HealthScore'] = (sb['OnTimeRate']*0.4 +
                         (sb['CSAT']/100)*0.3 +
                         (sb['Sales']/sb['Sales'].max())*0.2 +
                         (1-sb['TaskLoad']/sb['TaskLoad'].max())*0.1)

    # build region map once
    def flatten(n,p,rows):
        nm, tp = n['name'], n['type']
        npath = p+[nm] if tp in ('REGION','COMPANY') else p
        if tp=='STORE':
            rows.append({'Store':nm,
                         'Division':p[0] if len(p)>0 else 'Unknown',
                         'Region':p[1]   if len(p)>1 else 'Unknown'})
        for c in n.get('children',[]): flatten(c,npath,rows)

    with open('region_hierarchy.json') as f: h=json.load(f)
    rows=[]; flatten(h,[],rows)
    map_df=pd.DataFrame(rows)

    sb = sb.merge(map_df, on='Store', how='left')
    sb[['Division','Region']] = sb[['Division','Region']].fillna('Unknown')
    sb['TaskCount']=sb['TaskLoad']

    fig=px.treemap(sb,
        path=['Division','Region','Store'],
        values='TaskCount',
        color='HealthScore',
        color_continuous_scale='RdYlGn',
        hover_data=['HealthScore'],
        title='Health by Region → Store'
    )
    fig.update_layout(margin=dict(t=40,l=0,r=0,b=0))
    st.plotly_chart(fig,use_container_width=True)

# ─── Tab 3: Trends & Forecast ────────────────────────────────────────
with tab3:
    st.header("Weekly Compliance Trend")
    trend = (
        df.groupby('Week Start')
          .apply(lambda d: d.groupby('Task ID')['Days Before Due'].max().ge(0).mean())
          .rename("OnTimeRate")
          .reset_index()
          .sort_values('Week Start')
    )
    st.line_chart(trend.set_index('Week Start')['OnTimeRate'])

    st.subheader("Forecasted On-Time Rate Next Week")
    x = np.arange(len(trend)).reshape(-1,1); y = trend['OnTimeRate']
    if len(trend)>1:
        model = LinearRegression().fit(x, y); pred = model.predict([[len(trend)]])[0]
        st.metric("Forecasted On-Time %", f"{pred:.0%}")

    st.subheader("Upcoming Effort Requirement")
    base_rate = pred if len(trend)>1 else y.iloc[-1]
    fut = (filtered.groupby('Store')['Expected duration'].sum() * (1 - base_rate))
    st.bar_chart(fut.reset_index(name='Required Effort (hrs)').set_index('Store')['Required Effort (hrs)'])

# ─── Tab 4: Action & Accountability Tracker ─────────────────────────
with tab4:
    st.header("Action & Accountability Tracker")
    actions = over[['Store','Task ID','Task','Days Late']].copy()
    actions['Next Action Date'] = date.today() + timedelta(days=2)
    actions['Owner'] = "Store Manager"
    st.table(actions)

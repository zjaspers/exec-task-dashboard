import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from io import StringIO

# ─── Custom CSS for Cal AI-Inspired UI ───────────────────────────────
st.markdown("""
    <style>
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    .stPlotlyChart {
        width: 100% !important;
    }
    h1 {
        color: #1E88E5;
        font-size: 1.8rem;
        font-weight: bold;
    }
    h2, h3 {
        color: #333333;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stTabs {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
    }
    @media (max-width: 600px) {
        h1 { font-size: 1.4rem; }
        h2, h3 { font-size: 1rem; }
        .stMetric { font-size: 0.9rem; }
        .stButton > button { width: 100%; }
        .stPlotlyChart { height: 300px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────
def load_csv(file):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(file)

def load_xml(file, root_tag, ns):
    """Load XML file into a DataFrame."""
    tree = ET.parse(file)
    root = tree.getroot()
    data = []
    
    for elem in root.findall(f'./{ns}{root_tag}', {'x': ns}):
        record = {}
        for child in elem:
            tag = child.tag.replace(f'{{{ns}}}', '')
            if tag == 'descriptions':
                for desc in child:
                    if desc.attrib.get('lang') == 'en':
                        record['description'] = desc.text
            elif tag == 'ipMasks':
                record['ipMaskAddress'] = [ip.text for ip in child]
            elif tag == 'geolocationConfigs':
                record['geolocationConfigs'] = [(g.find(f'./{ns}latitude', {'x': ns}).text,
                                                g.find(f'./{ns}longitude', {'x': ns}).text,
                                                g.find(f'./{ns}radius', {'x': ns}).text)
                                               for g in child]
            elif tag == 'telephone':
                record['telephone'] = child.text
                record['telephone_type'] = child.attrib.get('type')
            else:
                record[tag] = child.text
        data.append(record)
    
    return pd.DataFrame(data)

def preprocess_tasks(df):
    """Preprocess task DataFrame."""
    df = df.copy()
    rename_map = {
        'Location name': 'Store',
        'Task name': 'Task',
        'Task status_x': 'Status',
        'End date': 'Due Date',
        'Date completed': 'Completed Date',
        'Task priority': 'Priority',
        'Assignee': 'Assignee'
    }
    df = df.rename(columns=rename_map)
    
    df['Due Date'] = pd.to_datetime(df['Due Date'], errors='coerce')
    df['Completed Date'] = pd.to_datetime(df['Completed Date'], errors='coerce')
    
    current_date = pd.Timestamp.now()
    df['Days Overdue'] = (current_date - df['Due Date']).dt.total_seconds() / (24 * 3600)
    df.loc[df['Status'] == 'Completed', 'Days Overdue'] = (
        (df['Completed Date'] - df['Due Date']).dt.total_seconds() / (24 * 3600)
    )
    
    df['Is Overdue'] = (df['Status'] != 'Completed') & (df['Due Date'] < current_date)
    return df

def validate_kpi_data(kpi_data):
    """Validate KPI DataFrame."""
    if kpi_data is None:
        return False, "KPI data is missing."
    
    column_mapping = {
        'Store': 'Store',
        'Sales vs Target': 'Sales vs Target (%)',
        'CSAT': 'CSAT Score',
        'Cleanliness Score': 'Cleanliness Score'
    }
    required_columns = ['Store', 'Sales vs Target', 'CSAT', 'Cleanliness Score']
    actual_columns = [column_mapping[col] for col in required_columns if col in column_mapping]
    missing_columns = [col for col in actual_columns if col not in kpi_data.columns]
    
    if missing_columns:
        return False, f"KPI data is missing columns: {', '.join(missing_columns)}."
    
    numeric_columns = ['Sales vs Target (%)', 'CSAT Score', 'Cleanliness Score']
    for col in numeric_columns:
        if col in kpi_data.columns and not pd.api.types.is_numeric_dtype(kpi_data[col]):
            return False, f"Column '{col}' must be numeric."
    
    return True, ""

def calculate_health_score(task_data, kpi_data=None):
    """Calculate store health score based on task compliance and KPIs."""
    compliance_data = (
        task_data.groupby('Store')
        .agg(
            ComplianceRate=('Is Overdue', lambda x: 1 - x.mean() if not x.empty else 0),
            OverdueTasks=('Is Overdue', 'sum'),
            HighPriorityOverdue=('Is Overdue', lambda x: sum(x & (task_data.loc[x.index, 'Priority'] == 'HIGH')))
        )
        .reset_index()
    )
    
    if kpi_data is not None:
        is_valid, error_msg = validate_kpi_data(kpi_data)
        if is_valid:
            try:
                kpi_data = kpi_data.rename(columns={
                    'Sales vs Target (%)': 'Sales vs Target',
                    'CSAT Score': 'CSAT',
                    'Cleanliness Score': 'Cleanliness Score'
                })
                kpi_health = kpi_data.groupby('Store').agg({
                    'Sales vs Target': 'mean',
                    'CSAT': 'mean',
                    'Cleanliness Score': 'mean'
                }).reset_index()
                compliance_data = compliance_data.merge(kpi_health, on='Store', how='left')
                
                compliance_data['Health Score'] = (
                    (compliance_data['ComplianceRate'] * 0.5) +
                    (compliance_data['Sales vs Target'].fillna(0) / 100 * 0.2) +
                    (compliance_data['CSAT'].fillna(50) / 100 * 0.2) +
                    (compliance_data['Cleanliness Score'].fillna(50) / 100 * 0.1)
                )
            except Exception as e:
                st.warning(f"Error processing KPI data: {str(e)}. Using compliance data only.")
                compliance_data['Health Score'] = compliance_data['ComplianceRate']
        else:
            st.warning(error_msg + " Using compliance data only.")
            compliance_data['Health Score'] = compliance_data['ComplianceRate']
    else:
        compliance_data['Health Score'] = compliance_data['ComplianceRate']
    
    return compliance_data

def forecast_completion(task_data, kpi_data=None):
    """Predict next week's completion rates."""
    forecast_data = []
    current_date = pd.Timestamp.now()
    
    for store in task_data['Store'].unique():
        store_tasks = task_data[task_data['Store'] == store]
        compliance_rate = 1 - store_tasks['Is Overdue'].mean()
        overdue_high_priority = sum(store_tasks['Is Overdue'] & (store_tasks['Priority'] == 'HIGH'))
        
        pred = max(0, compliance_rate * (1 - 0.1 * overdue_high_priority))
        
        if kpi_data is not None and store in kpi_data['Store'].values:
            is_valid, _ = validate_kpi_data(kpi_data)
            if is_valid:
                kpi_row = kpi_data[kpi_data['Store'] == store].rename(columns={
                    'Sales vs Target (%)': 'Sales vs Target',
                    'CSAT Score': 'CSAT',
                    'Cleanliness Score': 'Cleanliness Score'
                }).iloc[0]
                if kpi_row['CSAT'] < 70:
                    pred *= 0.9
                if kpi_row['Cleanliness Score'] < 70:
                    pred *= 0.95
                if kpi_row['Sales vs Target'] < 0:
                    pred *= 0.9
        
        risk_level = 'High' if pred < 0.5 else 'Medium' if pred < 0.75 else 'Low'
        forecast_data.append({
            'Store': store,
            'Forecasted Compliance Rate': pred,
            'Risk Level': risk_level
        })
    
    return pd.DataFrame(forecast_data)

def generate_recommendations(task_data, kpi_data, forecast_df, health_data, positions_df, locations_df):
    """Generate specific, actionable recommendations."""
    recommendations = []
    current_date = pd.Timestamp.now()
    
    location_region_map = locations_df.set_index('locationName')['region'].to_dict()
    
    overdue_tasks = task_data[task_data['Is Overdue']]
    if not overdue_tasks.empty:
        overdue_by_store = overdue_tasks.groupby('Store').agg({
            'Task': 'count',
            'Task category': lambda x: x.mode().iloc[0] if not x.empty and not x.isna().all() else 'Unknown',
            'Assignee': lambda x: x.mode().iloc[0] if not x.empty and not x.isna().all() and not x.mode().empty else None,
            'Priority': lambda x: 'HIGH' if 'HIGH' in x.values else 'MEDIUM' if 'MEDIUM' in x.values else 'LOW'
        }).reset_index().rename(columns={'Task': 'Overdue Count'})
        
        for _, row in overdue_by_store.iterrows():
            store = row['Store']
            task_category = row['Task category'].lower()
            overdue_count = row['Overdue Count']
            assignee = row['Assignee']
            priority = row['Priority']
            
            target_role = 'Store Manager' if priority == 'HIGH' or overdue_count > 3 else 'Sales Associate'
            target_role_id = positions_df[positions_df['description'] == target_role]['employeePositionId'].iloc[0] if target_role in positions_df['description'].values else '001'
            
            if overdue_count > 3:
                if assignee is None:
                    recommendations.append(
                        f"- **Visit {store}**: {overdue_count} {task_category} tasks are overdue with no assignee. "
                        f"Assign to {target_role} (ID: {target_role_id}) by {current_date + timedelta(days=3):%b %d}."
                    )
                else:
                    recommendations.append(
                        f"- **Visit {store}**: {overdue_count} {task_category} tasks are overdue. "
                        f"Check with {assignee} to clear the backlog by {current_date + timedelta(days=3):%b %d}."
                    )
            
            if assignee is None:
                recommendations.append(
                    f"- **Assign Task**: {overdue_count} overdue {task_category} tasks at {store} lack an assignee. "
                    f"Assign to {target_role} (ID: {target_role_id}) by {current_date + timedelta(days=2):%b %d}."
                )
            else:
                recommendations.append(
                    f"- **Reassign Task**: Move overdue {task_category} tasks at {store} to {target_role} (ID: {target_role_id})."
                )
    
    if kpi_data is not None:
        is_valid, _ = validate_kpi_data(kpi_data)
        if is_valid:
            kpi_data = kpi_data.rename(columns={
                'Sales vs Target (%)': 'Sales vs Target',
                'CSAT Score': 'CSAT',
                'Cleanliness Score': 'Cleanliness Score'
            })
            low_csat = kpi_data[kpi_data['CSAT'] < 70]['Store'].tolist()
            for store in low_csat:
                store_tasks = task_data[task_data['Store'] == store]
                if store_tasks['Is Overdue'].mean() > 0.2:
                    recommendations.append(
                        f"- **Add Training at {store}**: Low CSAT ({kpi_data[kpi_data['Store'] == store]['CSAT'].iloc[0]:.1f}). "
                        f"Assign customer service training to Sales Associates (ID: 001) by {current_date + timedelta(days=5):%b %d}."
                    )
            
            low_sales = kpi_data[kpi_data['Sales vs Target'] < 0]['Store'].tolist()
            for store in low_sales:
                recommendations.append(
                    f"- **Assign Planogram Task at {store}**: Sales are at {kpi_data[kpi_data['Store'] == store]['Sales vs Target'].iloc[0]}% of target. "
                    f"Add a planogram reset task to Sales Associates (ID: 001) by {current_date + timedelta(days=7):%b %d}."
                )
            
            low_cleanliness = kpi_data[kpi_data['Cleanliness Score'] < 70]['Store'].tolist()
            for store in low_cleanliness:
                recommendations.append(
                    f"- **Schedule Cleaning Task at {store}**: Low cleanliness score ({kpi_data[kpi_data['Store'] == store]['Cleanliness Score'].iloc[0]:.1f}). "
                    f"Assign a cleaning audit task to Store Manager (ID: 003) by {current_date + timedelta(days=4):%b %d}."
                )
    
    high_risk = forecast_df[forecast_df['Risk Level'] == 'High']['Store'].tolist()
    for store in high_risk:
        region = location_region_map.get(store, 'Unknown')
        recommendations.append(
            f"- **Alert Region Manager (ID: 009) for {region}**: {store} is at high risk of missing task deadlines next week. "
            f"Add labor or reassign high-priority tasks by {current_date + timedelta(days=2):%b %d}."
        )
    
    if not recommendations:
        recommendations.append(
            "- **Monitor Stores**: No critical issues detected. Review compliance weekly to stay on track."
        )
    
    return recommendations

# ─── Main Layout ────────────────────────────────────────────────────
st.title("WorkJam Store Performance Dashboard")
st.markdown("Track tasks, boost KPIs, and keep stores thriving. Upload your data to see compliance and get smart recommendations.")

# Upload Section
with st.expander("📂 Upload Data", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        task_file = st.file_uploader("Task CSV", type="csv", help="Upload better_task_report_full.csv")
    with col2:
        kpi_file = st.file_uploader("KPI CSV (optional)", type="csv", help="Upload better_store_kpis.csv")
    with col3:
        positions_file = st.file_uploader("Positions XML", type="xml", help="Upload employeepositionsforGPT.xml")
        locations_file = st.file_uploader("Locations XML", type="xml", help="Upload locationsforGPT.xml")
    
    if not task_file:
        st.info("Please upload a Task CSV to continue.")
        st.stop()

# Load and Preprocess Data
task_data = preprocess_tasks(load_csv(task_file))
kpi_data = load_csv(kpi_file) if kpi_file else None

# Load XML data (default to sample data if not uploaded)
if positions_file:
    positions_df = load_xml(positions_file, 'employeePosition', 'urn:employeePositions')
else:
    positions_xml = """
    <?xml version="1.0"?>
    <x:employeePositions xmlns:x="urn:employeePositions">
        <x:employeePosition>
            <employeePositionId>001</employeePositionId>
            <descriptions><description lang="en">Sales Associate</description></descriptions>
        </x:employeePosition>
        <x:employeePosition>
            <employeePositionId>003</employeePositionId>
            <descriptions><description lang="en">Store Manager</description></descriptions>
        </x:employeePosition>
        <x:employeePosition>
            <employeePositionId>009</employeePositionId>
            <descriptions><description lang="en">Region Manager</description></descriptions>
        </x:employeePosition>
    </x:employeePositions>
    """
    positions_df = load_xml(StringIO(positions_xml), 'employeePosition', 'urn:employeePositions')

if locations_file:
    locations_df = load_xml(locations_file, 'location', 'urn:locations')
else:
    locations_xml = """
    <?xml version="1.0"?>
    <x:locations xmlns:x="urn:locations">
        <x:location>
            <locationId>30477</locationId>
            <locationName>Aurora</locationName>
            <region>us-co</region>
        </x:location>
        <x:location>
            <locationId>10403</locationId>
            <locationName>La Plaza</locationName>
            <region>us-tx</region>
        </x:location>
    </x:locations>
    """
    locations_df = load_xml(StringIO(locations_xml), 'location', 'urn:locations')

health_data = calculate_health_score(task_data, kpi_data)
forecast_df = forecast_completion(task_data, kpi_data)

# Summary Section
st.header("At a Glance")
col1, col2, col3 = st.columns([1, 1, 1])
overdue_rate = task_data['Is Overdue'].mean()
col1.metric("Tasks Overdue", f"{overdue_rate:.0%}", delta=f"{overdue_rate-0.1:.0%}" if overdue_rate > 0.1 else None, delta_color="inverse")
col2.metric("Stores at Risk", sum(health_data['Health Score'] < 0.5), delta_color="inverse")
col3.metric("Avg. Store Health", f"{health_data['Health Score'].mean():.0%}", delta_color="normal")

# Issues Section
with st.expander("⚠️ Compliance Issues", expanded=True):
    st.subheader("Overdue Tasks by Store")
    overdue_by_store = task_data[task_data['Is Overdue']].groupby('Store').agg({
        'Task': 'count',
        'Task category': lambda x: x.mode().iloc[0] if not x.empty and not x.isna().all() else 'Unknown'
    }).reset_index().rename(columns={'Task': 'Overdue Tasks'})
    
    if not overdue_by_store.empty:
        fig_issues = px.bar(
            overdue_by_store,
            x='Store',
            y='Overdue Tasks',
            color='Task category',
            title="Overdue Tasks by Store",
            template="plotly_white",
            color_discrete_sequence=['#E53935', '#1E88E5', '#43A047']
        )
        fig_issues.update_layout(showlegend=True)
        st.plotly_chart(fig_issues, use_container_width=True)
        
        st.markdown("**Why It Matters**:")
        for _, row in overdue_by_store.iterrows():
            store = row['Store']
            store_kpis = kpi_data[kpi_data['Store'] == store] if kpi_data is not None else None
            if store_kpis is not None and not store_kpis.empty:
                is_valid, _ = validate_kpi_data(kpi_data)
                if is_valid:
                    kpi_row = store_kpis.rename(columns={
                        'Sales vs Target (%)': 'Sales vs Target',
                        'CSAT Score': 'CSAT',
                        'Cleanliness Score': 'Cleanliness Score'
                    }).iloc[0]
                    if kpi_row['Sales vs Target'] < 0:
                        st.markdown(f"- {store}: Overdue {row['Task category'].lower()} tasks may be hurting sales "
                                    f"({kpi_row['Sales vs Target']}% of target).")
                    if kpi_row['CSAT'] < 70:
                        st.markdown(f"- {store}: Poor compliance could be impacting CSAT ({kpi_row['CSAT']}).")
                    if kpi_row['Cleanliness Score'] < 70:
                        st.markdown(f"- {store}: Overdue tasks may contribute to low cleanliness "
                                    f"({kpi_row['Cleanliness Score']}%).")
    else:
        st.info("No overdue tasks detected.")

# Forecast Section
with st.expander("🔮 Next Week’s Outlook", expanded=False):
    st.subheader("Predicted Compliance Rates")
    fig_forecast = px.bar(
        forecast_df,
        x='Store',
        y='Forecasted Compliance Rate',
        color='Risk Level',
        title="Next Week’s Compliance Forecast",
        labels={"Forecasted Compliance Rate": "Compliance Rate (%)"},
        template="plotly_white",
        color_discrete_map={'Low': '#43A047', 'Medium': '#1E88E5', 'High': '#E53935'}
    )
    fig_forecast.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("**Risk Levels**:")
    for risk in ['High', 'Medium', 'Low']:
        stores = forecast_df[forecast_df['Risk Level'] == risk]['Store'].tolist()
        if stores:
            st.markdown(f"- **{risk} Risk**: {', '.join(stores)}")

# Recommendations Section
with st.expander("✅ Smart Recommendations", expanded=True):
    st.subheader("What to Do Next")
    recommendations = generate_recommendations(task_data, kpi_data, forecast_df, health_data, positions_df, locations_df)
    for rec in recommendations:
        st.markdown(rec)

# Drill-Down Tabs
st.header("Dive Deeper")
tab1, tab2 = st.tabs(["By Store", "By Task"])

with tab1:
    st.subheader("Store Performance Details")
    current_date = pd.Timestamp.now()
    
    for store in health_data['Store']:
        with st.expander(f"{store} Details"):
            store_tasks = task_data[task_data['Store'] == store]
            store_health = health_data[health_data['Store'] == store].iloc[0]
            
            # Store Information
            st.markdown("**Store Information**")
            store_info = locations_df[locations_df['locationName'] == store].iloc[0] if store in locations_df['locationName'].values else None
            if store_info is not None:
                address = f"{store_info['address1']}, {store_info.get('address2', '')}, {store_info['city']}, {store_info['region']}, {store_info['zipCode']}"
                st.write({
                    'Address': address.strip(', '),
                    'Region': store_info['region'],
                    'Contact': store_info['telephone']
                })
            else:
                st.write("Store location data not available.")
            
            # Key Metrics
            st.markdown("**Key Metrics**")
            col1, col2 = st.columns([1, 1])
            col1.metric("Compliance Rate", f"{store_health['ComplianceRate']:.0%}")
            col2.metric("Health Score", f"{store_health['Health Score']:.0%}")
            
            # Available Roles
            st.markdown("**Available Roles**")
            location_roles = positions_df[positions_df['locationType'] == 'LOCATION']['description'].tolist()
            st.write(", ".join(location_roles) if location_roles else "No location-specific roles available.")
            
            # Overdue Tasks
            overdue_tasks = store_tasks[store_tasks['Is Overdue']]
            if not overdue_tasks.empty:
                st.markdown("**Overdue Tasks**")
                overdue_summary = overdue_tasks.groupby(['Task', 'Task category', 'Priority', 'Assignee']).agg({
                    'Days Overdue': 'mean'
                }).reset_index().rename(columns={'Days Overdue': 'Avg Days Overdue'})
                overdue_summary['Avg Days Overdue'] = overdue_summary['Avg Days Overdue'].round(1)
                overdue_summary['Assignee'] = overdue_summary['Assignee'].fillna('Unassigned')
                st.dataframe(overdue_summary[['Task', 'Task category', 'Priority', 'Assignee', 'Avg Days Overdue']])
                
                # Chart: Overdue Tasks by Category
                overdue_by_category = overdue_tasks.groupby('Task category').size().reset_index(name='Count')
                fig_overdue = px.bar(
                    overdue_by_category,
                    x='Task category',
                    y='Count',
                    title="Overdue Tasks by Category",
                    template="plotly_white",
                    color='Task category',
                    color_discrete_sequence=['#E53935', '#1E88E5', '#43A047']
                )
                st.plotly_chart(fig_overdue, use_container_width=True)
                
                # Assignment Suggestions
                st.markdown("**Assignment Suggestions**")
                for _, row in overdue_summary.iterrows():
                    if row['Assignee'] == 'Unassigned':
                        target_role = 'Store Manager' if row['Priority'] == 'HIGH' else 'Sales Associate'
                        target_role_id = positions_df[positions_df['description'] == target_role]['employeePositionId'].iloc[0] if target_role in positions_df['description'].values else '001'
                        st.markdown(f"- Assign '{row['Task']}' to {target_role} (ID: {target_role_id}) by {current_date + timedelta(days=2):%b %d}.")
            
            # KPIs and Insights
            if kpi_data is not None and store in kpi_data['Store'].values:
                is_valid, _ = validate_kpi_data(kpi_data)
                if is_valid:
                    st.markdown("**KPIs**")
                    store_kpis = kpi_data[kpi_data['Store'] == store].rename(columns={
                        'Sales vs Target (%)': 'Sales vs Target',
                        'CSAT Score': 'CSAT',
                        'Cleanliness Score': 'Cleanliness Score'
                    }).iloc[0]
                    st.write({
                        'Sales vs Target': f"{store_kpis['Sales vs Target']}%",
                        'CSAT': f"{store_kpis['CSAT']:.1f}",
                        'Cleanliness Score': f"{store_kpis['Cleanliness Score']:.1f}"
                    })
                    
                    st.markdown("**KPI Insights**")
                    if store_kpis['CSAT'] < 70:
                        st.markdown(f"- Low CSAT ({store_kpis['CSAT']:.1f}): Assign customer service training to Sales Associates (ID: 001) by {current_date + timedelta(days=5):%b %d}.")
                    if store_kpis['Sales vs Target'] < 0:
                        st.markdown(f"- Low Sales ({store_kpis['Sales vs Target']}% of target): Add a planogram reset task to Sales Associates (ID: 001) by {current_date + timedelta(days=7):%b %d}.")
                    if store_kpis['Cleanliness Score'] < 70:
                        st.markdown(f"- Low Cleanliness ({store_kpis['Cleanliness Score']:.1f}): Assign a cleaning audit task to Store Manager (ID: 003) by {current_date + timedelta(days=4):%b %d}.")

with tab2:
    st.subheader("Task Compliance Across Stores")
    unique_tasks = task_data[['Task', 'Task category']].drop_duplicates()
    selected_task = st.selectbox("Select a Task", unique_tasks['Task'])
    
    task_compliance = task_data[task_data['Task'] == selected_task].groupby('Store').agg({
        'Is Overdue': lambda x: 1 - x.mean(),
        'Priority': 'first',
        'Assignee': 'first'
    }).reset_index().rename(columns={'Is Overdue': 'Compliance Rate'})
    
    fig_task = px.bar(
        task_compliance,
        x='Store',
        y='Compliance Rate',
        title=f"Compliance for {selected_task}",
        labels={'Compliance Rate': 'Compliance Rate (%)'},
        template="plotly_white",
        color='Compliance Rate',
        color_continuous_scale='RdYlGn'
    )
    fig_task.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_task, use_container_width=True)
    
    st.markdown("**Task Details**")
    task_details = task_data[task_data['Task'] == selected_task][['Task category', 'Priority', 'Assignee']].iloc[0]
    st.write({
        'Category': task_details['Task category'],
        'Priority': task_details['Priority'],
        'Assignee': task_details['Assignee'] if pd.notna(task_details['Assignee']) else 'Unassigned'
    })

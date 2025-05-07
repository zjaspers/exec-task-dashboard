# ─── Tab 1: Key Metrics & Recommendations ───────────────────────────
with tab1:
    st.header("Key Metrics")
    total_tasks   = filtered['Task ID'].nunique()
    on_time_count = filtered.groupby('Task ID')['Days Before Due'].max().ge(0).sum()
    avg_days      = filtered.groupby('Task ID')['Days Before Due'].mean().mean().round(1)
    overdue_count = total_tasks - on_time_count
    adhoc         = filtered.groupby('Task ID')['Store'].nunique().eq(1).sum()
    avg_csat      = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None

    cols = st.columns(6)
    with cols[0]: metric_card("Total Tasks", total_tasks)
    with cols[1]: metric_card("% On Time", f"{on_time_count/total_tasks:.0%}")
    with cols[2]: metric_card("Avg Days Before Due", avg_days)
    with cols[3]: metric_card("Overdue Tasks", overdue_count)
    with cols[4]: metric_card("Ad Hoc Tasks", adhoc)
    if avg_csat is not None:
        with cols[5]: metric_card("Avg CSAT", f"{avg_csat:.1f}")

    st.markdown("### Underperforming Tasks")
    # 1. Aggregate by Task ID (one row per task)
    underperf = (
        filtered
        .groupby('Task ID')
        .agg(
            Store      = ('Store','first'),
            Task       = ('Task name','first'),
            DaysLate   = ('Days Before Due','max')
        )
        .reset_index()
    )
    # 2. Only keep tasks that were late
    underperf = underperf[underperf['DaysLate'] < 0]
    # 3. Convert to positive days late and sort
    underperf['Days Late'] = -underperf['DaysLate']
    underperf = underperf[['Store','Task ID','Task','Days Late']]\
                  .sort_values('Days Late', ascending=False)\
                  .head(5)

    # 4. Display a compact table of the worst 5 tasks
    st.table(underperf)

    st.markdown("### Actionable Recommendations")
    st.markdown("- **Standardize your top break-fix tasks**: Create templates for the 5 tasks above so every store runs the same checklist.")
    st.markdown("- **Initiate manager check-ins**: Schedule a weekly 15-minute call with each store lead to review overdue tasks.")
    st.markdown("- **Empower region reps**: Give region managers visibility into store task boards and ask for escalation on any new overdue items.")

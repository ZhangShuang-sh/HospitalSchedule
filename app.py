"""
Hospital Shift Scheduling System - Streamlit UI
Rheumatology Department Monthly Shift Scheduler
"""

import streamlit as st
import pandas as pd
from datetime import date, datetime
import calendar

from utils import (
    get_default_staff_data,
    get_month_dates,
    get_weekends,
    create_schedule_dataframe,
    create_statistics_dataframe,
    validate_staff_data,
    get_shift_symbol,
    parse_date_list,
)
from scheduler_logic import Scheduler, create_staff_from_dataframe


def validate_staff_schedule_inputs(df, year: int, month: int) -> list:
    """
    Validate staff schedule inputs for logical conflicts.
    Returns a list of warning messages.
    """
    warnings = []

    for idx, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name:
            continue

        fixed_off_str = str(row.get("FixedOff", ""))
        fixed_on_str = str(row.get("FixedOn", ""))

        # Parse FixedOff dates
        fixed_off_dates = set()
        if fixed_off_str and fixed_off_str.strip() and fixed_off_str != "nan":
            fixed_off_dates = set(parse_date_list(fixed_off_str, year, month))

        # Parse FixedOn dates
        fixed_on_dates = set()
        if fixed_on_str and fixed_on_str.strip() and fixed_on_str.lower() != "nan":
            # Normalize separators
            fixed_on_str = fixed_on_str.replace("，", ",").replace("；", ",").replace("：", ":")
            for part in fixed_on_str.split(","):
                part = part.strip()
                if ":" in part:
                    try:
                        date_part = part.split(":")[0].strip()
                        if "-" in date_part:
                            from datetime import date as dt_date
                            d = dt_date.fromisoformat(date_part)
                        else:
                            from datetime import date as dt_date
                            d = dt_date(year, month, int(date_part))
                        fixed_on_dates.add(d)
                    except (ValueError, TypeError):
                        warnings.append(f"**{name}**: Invalid date format in FixedOn: '{part}'")

        # Check for conflicts: same date in both FixedOff and FixedOn
        conflicts = fixed_off_dates & fixed_on_dates
        if conflicts:
            conflict_dates = ", ".join(d.strftime("%m/%d") for d in sorted(conflicts))
            warnings.append(f"**{name}**: Date conflict - same date(s) in both FixedOff and FixedOn: {conflict_dates}")

        # Check for duplicate dates in FixedOn (same day assigned twice)
        if fixed_on_str and fixed_on_str.strip() and fixed_on_str.lower() != "nan":
            # Normalize separators (already normalized above, but be safe)
            normalized_on_str = fixed_on_str.replace("，", ",").replace("；", ",").replace("：", ":")
            on_dates_list = []
            for part in normalized_on_str.split(","):
                part = part.strip()
                if ":" in part:
                    try:
                        date_part = part.split(":")[0].strip()
                        if "-" in date_part:
                            from datetime import date as dt_date
                            d = dt_date.fromisoformat(date_part)
                        else:
                            from datetime import date as dt_date
                            d = dt_date(year, month, int(date_part))
                        on_dates_list.append(d)
                    except (ValueError, TypeError):
                        pass

            # Check duplicates
            seen = set()
            for d in on_dates_list:
                if d in seen:
                    warnings.append(f"**{name}**: Duplicate date in FixedOn: {d.strftime('%m/%d')}")
                seen.add(d)

        # Check CanDoNight=False but has Night shift in FixedOn
        can_do_night = bool(row.get("CanDoNight", True))
        if not can_do_night and fixed_on_str:
            if "Night" in fixed_on_str or "24h" in fixed_on_str:
                warnings.append(f"**{name}**: Has Night/24h in FixedOn but 'Night OK' is unchecked")

        # Check CanDo24h=False but has 24h shift in FixedOn
        can_do_24h = bool(row.get("CanDo24h", True))
        if not can_do_24h and fixed_on_str and "24h" in fixed_on_str:
            warnings.append(f"**{name}**: Has 24h in FixedOn but '24h OK' is unchecked")

    return warnings


# Page configuration
st.set_page_config(
    page_title="Hospital Shift Scheduler",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better table display
st.markdown("""
<style>
    .shift-day { background-color: #90EE90 !important; }
    .shift-night { background-color: #ADD8E6 !important; }
    .shift-24h { background-color: #FFB6C1 !important; }
    .weekend-col { background-color: #FFF3CD !important; }
    .holiday-col { background-color: #F8D7DA !important; }
    .stDataFrame { font-size: 12px; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "staff_df" not in st.session_state:
        st.session_state.staff_df = get_default_staff_data()
    if "schedule_generated" not in st.session_state:
        st.session_state.schedule_generated = False
    if "schedule_df" not in st.session_state:
        st.session_state.schedule_df = None
    if "stats_df" not in st.session_state:
        st.session_state.stats_df = None
    if "fairness_metrics" not in st.session_state:
        st.session_state.fairness_metrics = None
    if "coverage_summary" not in st.session_state:
        st.session_state.coverage_summary = None
    if "scheduler" not in st.session_state:
        st.session_state.scheduler = None
    if "reschedule_changes" not in st.session_state:
        st.session_state.reschedule_changes = []


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.header("Schedule Configuration")

    # Month and Year selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        year = st.number_input(
            "Year",
            min_value=2024,
            max_value=2030,
            value=2026,
            step=1,
        )
    with col2:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: calendar.month_name[x],
            index=0,
        )

    st.sidebar.divider()

    # Holidays input
    st.sidebar.subheader("Holidays")
    holidays_str = st.sidebar.text_input(
        "Holiday dates (comma-separated)",
        value="1",
        help="Enter day numbers separated by commas (e.g., 1,15,26) or full dates (e.g., 2026-01-01)",
    )

    st.sidebar.divider()

    # Shift configuration
    st.sidebar.subheader("Shift Settings")
    day_shifts = st.sidebar.number_input(
        "Day shifts per day",
        min_value=1,
        max_value=5,
        value=1,
        help="Number of Day shifts required each day. Auto-increases to 2 if surplus staff.",
    )
    night_shifts = st.sidebar.number_input(
        "Night shifts per day",
        min_value=1,
        max_value=3,
        value=1,
        help="Number of Night shifts required each day.",
    )

    st.sidebar.divider()

    # Legend
    st.sidebar.subheader("Shift Legend")
    st.sidebar.markdown("""
    - **D** = Day Shift (Weight: 1)
    - **N** = Night Shift (Weight: 1)
    - **24** = 24-Hour Shift (Weight: 2)
    """)

    return year, month, holidays_str, day_shifts, night_shifts


def render_staff_editor():
    """Render the staff data editor."""
    st.subheader("Staff Configuration")

    # File upload option
    uploaded_file = st.file_uploader(
        "Upload staff CSV (optional)",
        type=["csv"],
        help="CSV should have columns: Name, CanDoNight, CanDo24h, Availability, FixedOff, FixedOn",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            # Ensure required columns exist
            required_cols = ["Name"]
            if all(col in uploaded_df.columns for col in required_cols):
                # Add missing optional columns with defaults
                if "CanDoNight" not in uploaded_df.columns:
                    uploaded_df["CanDoNight"] = True
                if "CanDo24h" not in uploaded_df.columns:
                    uploaded_df["CanDo24h"] = True
                if "FixedOff" not in uploaded_df.columns:
                    uploaded_df["FixedOff"] = ""
                if "FixedOn" not in uploaded_df.columns:
                    uploaded_df["FixedOn"] = ""
                if "Role" not in uploaded_df.columns:
                    uploaded_df["Role"] = "Staff"

                st.session_state.staff_df = uploaded_df
                st.success("Staff data loaded from CSV!")
            else:
                st.error(f"CSV must contain columns: {required_cols}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    # Editable data table with form to prevent auto-refresh
    st.markdown("**Edit Staff Data:**")
    st.caption("Edit the table below, then click **Save Changes** to confirm. FixedOff: days off (e.g., '5,10,15' or '17-20'). FixedOn: preset shifts (e.g., '1:Day,25:Night').")

    # Column configuration for the editor
    column_config = {
        "Name": st.column_config.TextColumn("Name", required=True, width="medium"),
        "Role": st.column_config.SelectboxColumn(
            "Role",
            options=["Attending", "Fellow", "Resident", "Student", "Other"],
            width="small",
        ),
        "CanDoNight": st.column_config.CheckboxColumn("Night OK", width="small"),
        "CanDo24h": st.column_config.CheckboxColumn("24h OK", width="small"),
        "FixedOff": st.column_config.TextColumn(
            "Fixed Off",
            help="Days off: 5,10,15 or range 17-20. Off>=14 days → 50% shifts",
            width="large",
        ),
        "FixedOn": st.column_config.TextColumn(
            "Fixed On",
            help="Preset shifts: day:type (e.g., 1:Day,25:Night)",
            width="large",
        ),
    }

    # Use a form to batch edits and prevent auto-refresh
    with st.form("staff_editor_form"):
        edited_df = st.data_editor(
            st.session_state.staff_df,
            column_config=column_config,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )

        # Save button inside form
        col1, col2 = st.columns([1, 4])
        with col1:
            save_clicked = st.form_submit_button("Save Changes", type="primary")

        if save_clicked:
            st.session_state.staff_df = edited_df
            st.success("Changes saved!")

    # Show current staff count
    num_staff = len(st.session_state.staff_df[st.session_state.staff_df["Name"].notna() & (st.session_state.staff_df["Name"] != "")])
    st.caption(f"Staff count: {num_staff}")

    return num_staff > 0


def render_validation_warnings(year: int, month: int):
    """Display validation warnings for staff schedule inputs."""
    warnings = validate_staff_schedule_inputs(st.session_state.staff_df, year, month)

    if warnings:
        with st.expander(f"**Input Warnings ({len(warnings)})**", expanded=True):
            for warning in warnings:
                st.warning(warning)


def generate_schedule(year, month, holidays_str, day_shifts, night_shifts):
    """Generate the schedule based on current configuration."""
    # Parse holidays
    holidays = set(parse_date_list(holidays_str, year, month))

    # Create staff from DataFrame
    staff = create_staff_from_dataframe(st.session_state.staff_df, year, month)

    if not staff:
        st.error("No valid staff members configured!")
        return False

    # Debug: Show staff configurations with fixed dates and target ratio
    import calendar
    total_days = calendar.monthrange(year, month)[1]
    with st.expander("Debug: Staff Configuration Used", expanded=False):
        for person in staff:
            off_count = len(person.fixed_off_dates)
            ratio = person.get_target_ratio(total_days)
            ratio_text = "50%" if ratio == 0.5 else "100%"

            if person.fixed_off_dates or person.fixed_on_dates:
                st.write(f"**{person.name}** - {off_count} days off → {ratio_text} shifts")
                if person.fixed_off_dates:
                    off_dates = ", ".join(d.strftime("%m/%d") for d in sorted(person.fixed_off_dates))
                    st.write(f"  - Fixed Off: {off_dates}")
                if person.fixed_on_dates:
                    on_dates = ", ".join(f"{d.strftime('%m/%d')}:{s.value}" for d, s in sorted(person.fixed_on_dates.items()))
                    st.write(f"  - Fixed On: {on_dates}")

    # Create scheduler
    scheduler = Scheduler(
        year=year,
        month=month,
        staff=staff,
        holidays=holidays,
        day_shifts_per_day=day_shifts,
        night_shifts_per_day=night_shifts,
    )

    # Generate schedule with multiple attempts to find the fairest
    with st.spinner("Generating optimal schedule (trying 20 combinations)..."):
        success = scheduler.generate_schedule(num_attempts=20)

    if not success:
        st.warning("Could not achieve full coverage for all days. Review staff availability.")

    # Get results
    schedule_dict = scheduler.get_schedule_dict()
    staff_stats = scheduler.get_staff_stats()
    fairness = scheduler.get_fairness_metrics()
    coverage = scheduler.get_coverage_summary()

    # Create DataFrames
    dates = get_month_dates(year, month)
    staff_names = [p.name for p in staff]

    schedule_df = create_schedule_dataframe(staff_names, dates, schedule_dict)
    stats_df = create_statistics_dataframe(staff_stats)

    # Store in session state
    st.session_state.schedule_df = schedule_df
    st.session_state.stats_df = stats_df
    st.session_state.fairness_metrics = fairness
    st.session_state.coverage_summary = coverage
    st.session_state.schedule_generated = True
    st.session_state.current_year = year
    st.session_state.current_month = month
    st.session_state.holidays = holidays
    st.session_state.scheduler = scheduler  # Store scheduler for rescheduling
    st.session_state.reschedule_changes = []  # Clear previous changes

    return True


def style_schedule_cell(val, col_name, year, month, holidays):
    """Style individual cells in the schedule."""
    styles = []

    # Shift type colors
    if val == "Day":
        styles.append("background-color: #90EE90")  # Light green
    elif val == "Night":
        styles.append("background-color: #ADD8E6")  # Light blue
    elif val == "24h":
        styles.append("background-color: #FFB6C1")  # Light pink

    return "; ".join(styles) if styles else ""


def render_schedule_table():
    """Render the generated schedule table."""
    if st.session_state.schedule_df is None:
        return

    st.subheader("Generated Schedule")

    schedule_df = st.session_state.schedule_df.copy()
    year = st.session_state.current_year
    month = st.session_state.current_month
    holidays = st.session_state.holidays
    weekends = get_weekends(year, month)

    # Create styled DataFrame
    def highlight_shifts(val):
        if val == "Day":
            return "background-color: #90EE90; color: #000"
        elif val == "Night":
            return "background-color: #ADD8E6; color: #000"
        elif val == "24h":
            return "background-color: #FFB6C1; color: #000"
        return ""

    def highlight_columns(col):
        day_num = int(col.name)
        d = date(year, month, day_num)
        if d in holidays:
            return ["background-color: #F8D7DA"] * len(col)
        elif d in weekends:
            return ["background-color: #FFF3CD"] * len(col)
        return [""] * len(col)

    # Apply styling (compatible with pandas 2.0.x and 2.1+)
    try:
        # pandas >= 2.1.0
        styled_df = schedule_df.style.map(highlight_shifts).apply(highlight_columns, axis=0)
    except AttributeError:
        # pandas < 2.1.0
        styled_df = schedule_df.style.applymap(highlight_shifts).apply(highlight_columns, axis=0)

    # Add day names to column headers
    new_cols = {}
    for col in schedule_df.columns:
        day_num = int(col)
        d = date(year, month, day_num)
        day_name = d.strftime("%a")
        new_cols[col] = f"{col}\n{day_name}"

    # Display
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Legend
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("🟢 **Day Shift**")
    with col2:
        st.markdown("🔵 **Night Shift**")
    with col3:
        st.markdown("🔴 **24h Shift**")
    with col4:
        st.markdown("🟡 **Weekend**")
    with col5:
        st.markdown("🔶 **Holiday**")


def render_statistics():
    """Render statistics panel."""
    if st.session_state.stats_df is None:
        return

    st.subheader("Staff Statistics")

    stats_df = st.session_state.stats_df
    fairness = st.session_state.fairness_metrics

    # Calculate min-max ranges for fairness display
    weekend_min = stats_df["Weekend"].min()
    weekend_max = stats_df["Weekend"].max()
    night_min = stats_df["Night"].min()
    night_max = stats_df["Night"].max()
    day_min = stats_df["Day"].min()
    day_max = stats_df["Day"].max()

    # Fairness summary
    st.markdown("**Fairness Check (Max Diff ≤ 1 is ideal):**")
    col1, col2, col3 = st.columns(3)

    with col1:
        weekend_diff = weekend_max - weekend_min
        color = "green" if weekend_diff <= 1 else "red"
        st.markdown(f"Weekend: **:{color}[{weekend_min}-{weekend_max}]** (diff: {weekend_diff})")
    with col2:
        night_diff = night_max - night_min
        color = "green" if night_diff <= 1 else "red"
        st.markdown(f"Night: **:{color}[{night_min}-{night_max}]** (diff: {night_diff})")
    with col3:
        day_diff = day_max - day_min
        color = "green" if day_diff <= 1 else "red"
        st.markdown(f"Day: **:{color}[{day_min}-{day_max}]** (diff: {day_diff})")

    st.divider()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Avg Total",
            f"{fairness['total_shifts_mean']:.1f}",
            delta=f"StdDev: {fairness['total_shifts_stdev']:.2f}",
        )
    with col2:
        st.metric(
            "Avg Night",
            f"{fairness['night_shifts_mean']:.1f}",
            delta=f"StdDev: {fairness['night_shifts_stdev']:.2f}",
        )
    with col3:
        st.metric(
            "Avg Weekend",
            f"{fairness['weekend_shifts_mean']:.1f}",
            delta=f"StdDev: {fairness['weekend_shifts_stdev']:.2f}",
        )
    with col4:
        st.metric(
            "Avg Weighted",
            f"{fairness['weighted_mean']:.1f}",
            delta=f"StdDev: {fairness['weighted_stdev']:.2f}",
        )

    st.divider()

    # Full statistics table
    st.dataframe(
        stats_df.style.highlight_max(
            subset=["Total", "Night", "Weekend"],
            color="#FFCCCB",
        ).highlight_min(
            subset=["Total", "Night", "Weekend"],
            color="#90EE90",
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_coverage_summary():
    """Render daily coverage summary."""
    if st.session_state.coverage_summary is None:
        return

    with st.expander("Daily Coverage Details"):
        coverage = st.session_state.coverage_summary

        coverage_data = []
        for day_info in coverage:
            coverage_data.append({
                "Date": day_info["date"].strftime("%Y-%m-%d"),
                "Day": day_info["day_of_week"],
                "Weekend": "Yes" if day_info["is_weekend"] else "",
                "Holiday": "Yes" if day_info["is_holiday"] else "",
                "Day Staff": ", ".join(day_info["day_staff"]),
                "Night Staff": ", ".join(day_info["night_staff"]),
                "Day #": day_info["day_coverage"],
                "Night #": day_info["night_coverage"],
            })

        coverage_df = pd.DataFrame(coverage_data)
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)


def render_export_options():
    """Render export/download options."""
    if st.session_state.schedule_df is None:
        return

    st.subheader("Export")

    col1, col2 = st.columns(2)

    with col1:
        # CSV export
        csv_data = st.session_state.schedule_df.to_csv()
        st.download_button(
            label="Download Schedule (CSV)",
            data=csv_data,
            file_name=f"schedule_{st.session_state.current_year}_{st.session_state.current_month:02d}.csv",
            mime="text/csv",
        )

    with col2:
        # Stats CSV export
        stats_csv = st.session_state.stats_df.to_csv(index=False)
        st.download_button(
            label="Download Statistics (CSV)",
            data=stats_csv,
            file_name=f"stats_{st.session_state.current_year}_{st.session_state.current_month:02d}.csv",
            mime="text/csv",
        )


def render_temporary_leave():
    """Render temporary leave input and rescheduling section."""
    if st.session_state.scheduler is None:
        return

    st.divider()
    st.subheader("Temporary Leave & Rescheduling")
    st.caption("If someone needs to take temporary leave, enter their info below to reschedule their shifts.")

    scheduler = st.session_state.scheduler
    year = st.session_state.current_year
    month = st.session_state.current_month

    # Get list of staff names for selection
    staff_names = list(scheduler.staff.keys())

    col1, col2 = st.columns([1, 2])

    with col1:
        leave_person = st.selectbox(
            "Person taking leave",
            options=[""] + staff_names,
            index=0,
            help="Select the person who needs to take temporary leave",
        )

    with col2:
        leave_dates_str = st.text_input(
            "Leave dates",
            placeholder="e.g., 15,16,17 or 15-17",
            help="Enter dates (supports ranges like 15-17)",
        )

    if st.button("Reschedule", type="primary", disabled=not leave_person or not leave_dates_str):
        # Parse leave dates
        leave_dates = set(parse_date_list(leave_dates_str, year, month))

        if not leave_dates:
            st.error("Please enter valid leave dates.")
            return

        # Perform rescheduling
        changes = scheduler.reschedule_for_leave(leave_person, leave_dates)

        if changes:
            st.session_state.reschedule_changes = changes

            # Update the schedule DataFrame
            schedule_dict = scheduler.get_schedule_dict()
            staff_stats = scheduler.get_staff_stats()
            fairness = scheduler.get_fairness_metrics()
            coverage = scheduler.get_coverage_summary()

            dates = get_month_dates(year, month)
            staff_names_list = list(scheduler.staff.keys())

            st.session_state.schedule_df = create_schedule_dataframe(staff_names_list, dates, schedule_dict)
            st.session_state.stats_df = create_statistics_dataframe(staff_stats)
            st.session_state.fairness_metrics = fairness
            st.session_state.coverage_summary = coverage

            st.success(f"Rescheduled {len(changes)} shift(s)!")
            st.rerun()
        else:
            st.info(f"{leave_person} has no shifts on the specified dates.")

    # Show reschedule changes if any
    if st.session_state.reschedule_changes:
        st.markdown("**Recent Rescheduling Changes:**")
        for change in st.session_state.reschedule_changes:
            d = change["date"]
            shift = change["shift_type"]
            from_name = change["from"]
            to_name = change["to"]
            warning = change.get("warning")

            if warning:
                st.warning(f"- {d.strftime('%m/%d')} ({shift}): {warning}")
            elif to_name and to_name != from_name:
                st.success(f"- {d.strftime('%m/%d')} ({shift}): {from_name} → **{to_name}**")
            elif to_name == from_name:
                st.warning(f"- {d.strftime('%m/%d')} ({shift}): No replacement available, {from_name} must work")
            else:
                st.error(f"- {d.strftime('%m/%d')} ({shift}): {from_name} → **No replacement found!**")


def main():
    """Main application entry point."""
    st.title("Hospital Shift Scheduling System")
    st.caption("Rheumatology Department - Monthly Shift Scheduler")

    # Initialize session state
    init_session_state()

    # Sidebar configuration
    year, month, holidays_str, day_shifts, night_shifts = render_sidebar()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Staff Setup", "Schedule", "Statistics"])

    with tab1:
        staff_valid = render_staff_editor()

        st.divider()

        # Generate button
        if st.button(
            "Generate Schedule",
            type="primary",
            disabled=not staff_valid,
            use_container_width=True,
        ):
            # Show validation warnings before generating
            render_validation_warnings(year, month)

            if generate_schedule(year, month, holidays_str, day_shifts, night_shifts):
                st.success("Schedule generated successfully!")
                st.balloons()

    with tab2:
        if st.session_state.schedule_generated:
            render_schedule_table()
            render_temporary_leave()
            render_coverage_summary()
            render_export_options()
        else:
            st.info("Configure staff and click 'Generate Schedule' to create the monthly schedule.")

    with tab3:
        if st.session_state.schedule_generated:
            render_statistics()
        else:
            st.info("Generate a schedule to view statistics.")


if __name__ == "__main__":
    main()

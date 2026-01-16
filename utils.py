"""
Utility functions for Hospital Shift Scheduling System.
Handles date operations, data generation, and helper functions.
"""

import calendar
from datetime import date, timedelta
from typing import List, Tuple, Set
import pandas as pd


def get_month_dates(year: int, month: int) -> List[date]:
    """Get all dates in a given month."""
    num_days = calendar.monthrange(year, month)[1]
    return [date(year, month, day) for day in range(1, num_days + 1)]


def get_weekends(year: int, month: int) -> Set[date]:
    """Get all weekend dates (Saturday and Sunday) in a month."""
    weekends = set()
    for d in get_month_dates(year, month):
        if d.weekday() in (5, 6):  # Saturday=5, Sunday=6
            weekends.add(d)
    return weekends


def get_thursdays(year: int, month: int) -> List[date]:
    """Get all Thursday dates in a month."""
    return [d for d in get_month_dates(year, month) if d.weekday() == 3]


def is_weekend(d: date) -> bool:
    """Check if a date is a weekend."""
    return d.weekday() in (5, 6)


def is_thursday(d: date) -> bool:
    """Check if a date is Thursday."""
    return d.weekday() == 3


def get_day_name(d: date) -> str:
    """Get abbreviated day name."""
    return d.strftime("%a")


def format_date_header(d: date) -> str:
    """Format date for table header."""
    return f"{d.day}\n{get_day_name(d)}"


def days_between(d1: date, d2: date) -> int:
    """Calculate absolute days between two dates."""
    return abs((d2 - d1).days)


def get_default_staff_data() -> pd.DataFrame:
    """Generate default staff data for demo purposes."""
    data = {
        "Name": [
            "Dr. Smith",
            "Dr. Johnson",
            "Dr. Williams",
            "Dr. Brown",
            "Dr. Jones",
            "Dr. Garcia",
            "Dr. Miller",
            "Dr. Davis",
            "Student A",
            "Student B",
        ],
        "Role": [
            "Attending",
            "Attending",
            "Attending",
            "Fellow",
            "Fellow",
            "Resident",
            "Resident",
            "Resident",
            "Student",
            "Student",
        ],
        "CanDoNight": [True, True, True, True, True, True, True, True, False, False],
        "CanDo24h": [True, True, False, True, True, False, False, False, False, False],
        "FixedOff": ["", "", "", "", "16-31", "", "", "8-14", "", ""],
        "FixedOn": ["", "", "", "", "", "", "", "", "", ""],
    }
    return pd.DataFrame(data)


def parse_date_list(date_str: str, year: int, month: int) -> List[date]:
    """
    Parse a comma-separated string of dates.
    Supports formats:
    - "1,2,3" - individual days
    - "2026-01-01,2026-01-02" - full dates
    - "17-20" - date range (days 17, 18, 19, 20)
    Also supports Chinese comma (，) and semicolon (;) as separators.
    """
    if not date_str or date_str.strip() == "":
        return []

    # Handle "nan" string from pandas
    if date_str.strip().lower() == "nan":
        return []

    # Normalize separators: replace Chinese comma and semicolon with English comma
    normalized = date_str.replace("，", ",").replace(";", ",").replace("；", ",")

    dates = []
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                # Check if it's a date range like "17-20" (two numbers)
                parts = part.split("-")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    # Date range format: start-end (e.g., "17-20")
                    start_day = int(parts[0])
                    end_day = int(parts[1])
                    for day in range(start_day, end_day + 1):
                        try:
                            d = date(year, month, day)
                            dates.append(d)
                        except ValueError:
                            continue
                else:
                    # Full date format: YYYY-MM-DD
                    d = date.fromisoformat(part)
                    dates.append(d)
            else:
                # Day only
                d = date(year, month, int(part))
                dates.append(d)
        except (ValueError, TypeError):
            continue
    return dates


def calculate_target_shifts(
    num_staff: int,
    num_days: int,
    day_shifts_per_day: int = 1,
    night_shifts_per_day: int = 1,
) -> float:
    """Calculate target shifts per person."""
    total_shifts = (day_shifts_per_day + night_shifts_per_day) * num_days
    return total_shifts / num_staff if num_staff > 0 else 0


def should_increase_day_shifts(avg_shifts: float) -> bool:
    """Determine if we should increase daily Day shifts due to surplus staff."""
    return avg_shifts < 4


def create_schedule_dataframe(
    staff_names: List[str], dates: List[date], schedule: dict
) -> pd.DataFrame:
    """
    Create a DataFrame from schedule dictionary.

    Args:
        staff_names: List of staff names
        dates: List of dates in the month
        schedule: Dict mapping (person, date) -> shift_type

    Returns:
        DataFrame with staff as rows and dates as columns
    """
    data = {}
    for d in dates:
        col_name = f"{d.day}"
        data[col_name] = []
        for name in staff_names:
            shift = schedule.get((name, d), "")
            data[col_name].append(shift)

    df = pd.DataFrame(data, index=staff_names)
    df.index.name = "Staff"
    return df


def create_statistics_dataframe(staff_stats: dict) -> pd.DataFrame:
    """
    Create statistics DataFrame from staff stats.

    Args:
        staff_stats: Dict with staff statistics

    Returns:
        DataFrame with statistics
    """
    rows = []
    for name, stats in staff_stats.items():
        rows.append({
            "Name": name,
            "Target": stats.get("target_shifts", 0),
            "Total": stats.get("total_shifts", 0),
            "Day": stats.get("day_shifts", 0),
            "Night": stats.get("night_shifts", 0),
            "24h": stats.get("24h_shifts", 0),
            "Weekend": stats.get("weekend_shifts", 0),
            "Holiday": stats.get("holiday_shifts", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Name").reset_index(drop=True)
    return df


def get_shift_symbol(shift_type: str) -> str:
    """Get display symbol for shift type."""
    symbols = {
        "Day": "D",
        "Night": "N",
        "24h": "24",
    }
    return symbols.get(shift_type, "")


def get_shift_color(shift_type: str) -> str:
    """Get color code for shift type (for styling)."""
    colors = {
        "Day": "#90EE90",      # Light green
        "Night": "#ADD8E6",    # Light blue
        "24h": "#FFB6C1",      # Light pink
    }
    return colors.get(shift_type, "#FFFFFF")


def validate_staff_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate staff DataFrame has required columns.

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_cols = ["Name", "CanDoNight", "CanDo24h"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    if df.empty:
        return False, "Staff list is empty"

    if df["Name"].duplicated().any():
        return False, "Duplicate staff names found"

    return True, ""


def export_schedule_to_csv(schedule_df: pd.DataFrame, stats_df: pd.DataFrame) -> str:
    """
    Export schedule and stats to CSV string.
    """
    output = "=== SHIFT SCHEDULE ===\n"
    output += schedule_df.to_csv()
    output += "\n\n=== STATISTICS ===\n"
    output += stats_df.to_csv(index=False)
    return output

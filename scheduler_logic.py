"""
Core Scheduler Logic for Hospital Shift Scheduling System.
Implements the scheduling algorithm with all constraints and fairness rules.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import random


class ShiftType(Enum):
    DAY = "Day"
    NIGHT = "Night"
    FULL_24H = "24h"




SHIFT_WEIGHTS = {
    ShiftType.DAY: 1,
    ShiftType.NIGHT: 1,
    ShiftType.FULL_24H: 2,
}


# Threshold for determining half-month (days off >= this means 50% target)
HALF_MONTH_THRESHOLD = 14


@dataclass
class Person:
    """Represents a staff member with their constraints and statistics."""
    name: str
    can_do_night: bool = True
    can_do_24h: bool = True
    fixed_off_dates: Set[date] = field(default_factory=set)
    fixed_on_dates: Dict[date, ShiftType] = field(default_factory=dict)

    # Tracking stats
    total_shifts: int = 0
    day_shifts: int = 0
    night_shifts: int = 0
    shifts_24h: int = 0
    weekend_shifts: int = 0
    holiday_shifts: int = 0
    weighted_total: float = 0

    # Target shifts (calculated by scheduler)
    target_shifts: float = 0

    # History tracking
    last_day_shift: Optional[date] = None
    last_night_shift: Optional[date] = None
    assigned_dates: Set[date] = field(default_factory=set)

    def is_available_on(self, d: date) -> bool:
        """Check if person is available on a specific date."""
        return d not in self.fixed_off_dates

    def reset_monthly_stats(self):
        """Reset stats for a new month."""
        self.total_shifts = 0
        self.day_shifts = 0
        self.night_shifts = 0
        self.shifts_24h = 0
        self.weekend_shifts = 0
        self.holiday_shifts = 0
        self.weighted_total = 0
        self.last_day_shift = None
        self.last_night_shift = None
        self.assigned_dates = set()

    def get_target_ratio(self, total_days: int) -> float:
        """
        Get target shift ratio based on fixed off days.
        - Off >= 14 days (half month): 50% of normal shifts
        - Off < 14 days (about a week): 100% of normal shifts
        """
        off_days = len(self.fixed_off_dates)
        if off_days >= HALF_MONTH_THRESHOLD:
            return 0.5
        return 1.0

    def get_stats_dict(self) -> dict:
        """Return stats as dictionary."""
        return {
            "target_shifts": round(self.target_shifts, 1),
            "total_shifts": self.total_shifts,
            "day_shifts": self.day_shifts,
            "night_shifts": self.night_shifts,
            "24h_shifts": self.shifts_24h,
            "weekend_shifts": self.weekend_shifts,
            "holiday_shifts": self.holiday_shifts,
            "weighted_total": self.weighted_total,
        }


class Scheduler:
    """
    Main scheduler class implementing the hospital shift scheduling algorithm.
    Uses a greedy/scoring approach with constraint satisfaction.
    """

    # Gap constraints
    MIN_NIGHT_TO_NIGHT_GAP = 5      # Normal: minimum 5 days between night shifts
    EMERGENCY_NIGHT_TO_NIGHT_GAP = 3  # Fallback: minimum 3 days if not enough staff
    MIN_DAY_TO_DAY_GAP = 3          # Minimum 3 days between day shifts
    MIN_NIGHT_TO_DAY_GAP = 3        # After night, must wait 3 days for day shift

    def __init__(
        self,
        year: int,
        month: int,
        staff: List[Person],
        holidays: Set[date] = None,
        day_shifts_per_day: int = 1,
        night_shifts_per_day: int = 1,
    ):
        self.year = year
        self.month = month
        self.staff = {p.name: p for p in staff}
        self.holidays = holidays or set()
        self.day_shifts_per_day = day_shifts_per_day
        self.night_shifts_per_day = night_shifts_per_day

        # Generate dates for the month
        self._generate_month_dates()

        # Schedule storage: (person_name, date) -> ShiftType
        self.schedule: Dict[Tuple[str, date], ShiftType] = {}

        # Calculate targets
        self._calculate_targets()

    def _generate_month_dates(self):
        """Generate all dates in the scheduling month."""
        import calendar
        num_days = calendar.monthrange(self.year, self.month)[1]
        self.dates = [date(self.year, self.month, d) for d in range(1, num_days + 1)]
        self.weekends = {d for d in self.dates if d.weekday() in (5, 6)}
        self.thursdays = [d for d in self.dates if d.weekday() == 3]

    def _calculate_targets(self):
        """Calculate target shifts per person based on staff count and fixed off days."""
        # Exclude holidays from calculation (holidays are assigned via FixedOn lottery)
        working_days = len([d for d in self.dates if d not in self.holidays])
        total_days = len(self.dates)
        total_day_shifts = self.day_shifts_per_day * working_days
        total_night_shifts = self.night_shifts_per_day * working_days
        total_shifts = total_day_shifts + total_night_shifts

        # Calculate effective staff count (accounting for fixed off days)
        # People with >= 14 days off count as 0.5, others count as 1.0
        effective_staff = sum(p.get_target_ratio(total_days) for p in self.staff.values())

        if effective_staff > 0:
            self.avg_shifts_per_person = total_shifts / effective_staff
        else:
            self.avg_shifts_per_person = 0

        # Check surplus logic
        if self.avg_shifts_per_person < 4 and self.day_shifts_per_day == 1:
            self.day_shifts_per_day = 2
            self._calculate_targets()  # Recalculate with updated day shifts

        # Set individual targets (round to nearest integer for half-month people)
        for person in self.staff.values():
            ratio = person.get_target_ratio(total_days)
            raw_target = self.avg_shifts_per_person * ratio
            # Round 0.5 to 1 (四舍五入)
            person.target_shifts = round(raw_target)

    def is_holiday(self, d: date) -> bool:
        """Check if date is a holiday."""
        return d in self.holidays

    def is_weekend(self, d: date) -> bool:
        """Check if date is a weekend."""
        return d in self.weekends

    def is_weekend_or_holiday(self, d: date) -> bool:
        """Check if date is weekend or holiday."""
        return self.is_weekend(d) or self.is_holiday(d)

    def _get_weekend_pair(self, d: date) -> Optional[date]:
        """Get the other day of the same weekend (Sat->Sun or Sun->Sat)."""
        if d.weekday() == 5:  # Saturday
            return d + timedelta(days=1)  # Sunday
        elif d.weekday() == 6:  # Sunday
            return d - timedelta(days=1)  # Saturday
        return None

    def _get_thursday_for_weekend(self, d: date) -> Optional[date]:
        """Get the Thursday before a weekend date."""
        if d.weekday() == 5:  # Saturday
            return d - timedelta(days=2)  # Thursday
        elif d.weekday() == 6:  # Sunday
            return d - timedelta(days=3)  # Thursday
        return None

    def _has_thursday_night_this_week(self, person: Person, weekend_date: date) -> bool:
        """Check if person has Thursday night shift for the weekend's week."""
        thursday = self._get_thursday_for_weekend(weekend_date)
        if thursday and thursday in person.assigned_dates:
            shift = self.schedule.get((person.name, thursday))
            if shift in (ShiftType.NIGHT, ShiftType.FULL_24H):
                return True
        return False

    def _basic_can_assign(self, person: Person, d: date, shift_type: ShiftType) -> bool:
        """
        Basic assignment check without fairness constraints.
        Used to check if other candidates are available for fairness comparison.
        """
        # Check availability period
        if not person.is_available_on(d):
            return False

        # Check fixed off dates
        if d in person.fixed_off_dates:
            return False

        # Check if already assigned this date
        if d in person.assigned_dates:
            return False

        # Check capability
        if shift_type == ShiftType.NIGHT and not person.can_do_night:
            return False
        if shift_type == ShiftType.FULL_24H and not person.can_do_24h:
            return False

        # Check gap rules - Night shifts (NIGHT or 24H)
        # Check against ALL existing night shifts to handle non-chronological assignment
        if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    gap = abs((d - assigned_date).days)
                    if gap > 0 and gap < self.MIN_NIGHT_TO_NIGHT_GAP:
                        return False
            # Also check: if assigning a night shift, are there day shifts within
            # the next MIN_NIGHT_TO_DAY_GAP days that would be violated?
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.DAY, ShiftType.FULL_24H):
                    gap = (assigned_date - d).days  # day_date - night_date
                    if gap > 0 and gap < self.MIN_NIGHT_TO_DAY_GAP:
                        return False  # Day shift too soon after this night shift

        # Check gap rules - Day shifts (DAY or 24H)
        if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
            # Night-to-Day gap: no day shift within MIN_NIGHT_TO_DAY_GAP days after night
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    gap = (d - assigned_date).days
                    if gap > 0 and gap < self.MIN_NIGHT_TO_DAY_GAP:
                        return False
            # Day-to-Day gap: check against ALL existing day shifts
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.DAY, ShiftType.FULL_24H):
                    gap = abs((d - assigned_date).days)
                    if gap > 0 and gap < self.MIN_DAY_TO_DAY_GAP:
                        return False

        # Weekend constraint: max 1 shift per weekend per person
        if self.is_weekend(d):
            other_day = self._get_weekend_pair(d)
            if other_day and other_day in person.assigned_dates:
                return False  # Already has a shift on the other day of this weekend

            # Thursday night compensation: if person has Thursday night, skip this weekend
            if self._has_thursday_night_this_week(person, d):
                return False

        return True

    def _can_assign_shift(
        self, person: Person, d: date, shift_type: ShiftType
    ) -> Tuple[bool, str]:
        """
        Check if a person can be assigned a shift on a given date.
        Returns (can_assign, reason).
        """
        # Check if person is available on this date (unavailable period)
        if not person.is_available_on(d):
            return False, "Not available (unavailable period)"

        # Check fixed off dates
        if d in person.fixed_off_dates:
            return False, "Fixed off date"

        # Check if already assigned this date
        if d in person.assigned_dates:
            return False, "Already assigned"

        # Check capability
        if shift_type == ShiftType.NIGHT and not person.can_do_night:
            return False, "Cannot do night shifts"
        if shift_type == ShiftType.FULL_24H and not person.can_do_24h:
            return False, "Cannot do 24h shifts"

        # Check gap rules - Night shifts (NIGHT or 24H)
        # Check against ALL existing night shifts to handle non-chronological assignment
        if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    gap = abs((d - assigned_date).days)
                    if gap > 0 and gap < self.MIN_NIGHT_TO_NIGHT_GAP:
                        return False, f"Night gap too short ({gap} days)"
            # Also check: if assigning a night shift, are there day shifts within
            # the next MIN_NIGHT_TO_DAY_GAP days that would be violated?
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.DAY, ShiftType.FULL_24H):
                    gap = (assigned_date - d).days  # day_date - night_date
                    if gap > 0 and gap < self.MIN_NIGHT_TO_DAY_GAP:
                        return False, f"Existing day shift too close ({gap} days after)"

        # Check gap rules - Day shifts (DAY or 24H)
        if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
            # Night-to-Day gap: no day shift within MIN_NIGHT_TO_DAY_GAP days after night
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    gap = (d - assigned_date).days
                    if gap > 0 and gap < self.MIN_NIGHT_TO_DAY_GAP:
                        return False, f"Night-to-Day gap too short ({gap} days)"
            # Day-to-Day gap: check against ALL existing day shifts
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.DAY, ShiftType.FULL_24H):
                    gap = abs((d - assigned_date).days)
                    if gap > 0 and gap < self.MIN_DAY_TO_DAY_GAP:
                        return False, f"Day gap too short ({gap} days)"

        # Weekend constraint: max 1 shift per weekend per person
        if self.is_weekend(d):
            other_day = self._get_weekend_pair(d)
            if other_day and other_day in person.assigned_dates:
                return False, "Already has shift on other day of this weekend"

            # Thursday night compensation: if person has Thursday night, skip this weekend
            if self._has_thursday_night_this_week(person, d):
                return False, "Has Thursday night - weekend off as compensation"

        # HARD CONSTRAINT: Night shift fairness - max difference of 1
        # Only apply to those who can do night shifts
        if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
            night_capable = [p for p in self.staff.values() if p.can_do_night]
            if night_capable:
                total_days = len(self.dates)

                # Normalize for target ratio comparison
                def get_norm_night(p):
                    ratio = p.get_target_ratio(total_days)
                    return p.night_shifts * 2 if ratio == 0.5 else p.night_shifts

                person_norm = get_norm_night(person)
                min_norm = min(get_norm_night(p) for p in night_capable)

                # Block if this person has more normalized night shifts
                if person_norm > min_norm:
                    others_with_fewer = [
                        p for p in night_capable
                        if p.name != person.name
                        and get_norm_night(p) <= min_norm
                        and self._basic_can_assign(p, d, shift_type)
                    ]
                    if others_with_fewer:
                        return False, "Night fairness: others have fewer night shifts"

        # HARD CONSTRAINT: Weekend fairness - max difference of 1
        # Only compare among "comparable" people (exclude holiday workers from comparison)
        if self.is_weekend(d):
            # Block holiday workers from weekend if others available (highest priority)
            if person.holiday_shifts > 0:
                others_no_holiday = [
                    p for p in self.staff.values()
                    if p.name != person.name
                    and p.holiday_shifts == 0
                    and self._basic_can_assign(p, d, shift_type)
                ]
                if others_no_holiday:
                    return False, "Holiday worker should not work weekend"

            # Weekend fairness only among non-holiday workers
            # (holiday workers are expected to have fewer weekends - that's fair compensation)
            comparable_staff = [p for p in self.staff.values() if p.holiday_shifts == 0]
            if comparable_staff and person.holiday_shifts == 0:
                min_weekend = min(p.weekend_shifts for p in comparable_staff)

                # Block if this person already has more weekend shifts than minimum
                if person.weekend_shifts > min_weekend:
                    others_with_fewer = [
                        p for p in comparable_staff
                        if p.name != person.name
                        and p.weekend_shifts <= min_weekend
                        and self._basic_can_assign(p, d, shift_type)
                    ]
                    if others_with_fewer:
                        return False, "Weekend fairness: others have fewer weekend shifts"

                # Block if assigning would create difference > 1
                if person.weekend_shifts >= min_weekend + 1:
                    others_available = [
                        p for p in comparable_staff
                        if p.name != person.name
                        and p.weekend_shifts < person.weekend_shifts
                        and self._basic_can_assign(p, d, shift_type)
                    ]
                    if others_available:
                        return False, "Weekend fairness: would exceed max diff of 1"

        # Check if person has reached their target while others haven't
        total_days = len(self.dates)
        person_ratio = person.get_target_ratio(total_days)
        person_target = self.avg_shifts_per_person * person_ratio

        # If person has exceeded target by 1+, check if others are below target
        if person.total_shifts >= person_target + 1:
            others_below_target = any(
                p.total_shifts < self.avg_shifts_per_person * p.get_target_ratio(total_days)
                and p.name != person.name
                and self._basic_can_assign(p, d, shift_type)
                for p in self.staff.values()
            )
            if others_below_target:
                return False, "Exceeded target, others need shifts"

        return True, "OK"

    def _calculate_assignment_score(
        self, person: Person, d: date, shift_type: ShiftType
    ) -> float:
        """
        Calculate a score for assigning a shift to a person.
        Higher score = more preferred assignment.
        Prioritizes fairness to keep differences within 1 shift.
        """
        score = 0.0
        total_days = len(self.dates)

        # Get person's target ratio (0.5 for half-month, 1.0 for others)
        person_ratio = person.get_target_ratio(total_days)

        # Calculate normalized shifts (adjusted for target ratio)
        # This ensures Jones with 50% target is compared fairly
        def get_normalized_shifts(p: Person) -> float:
            ratio = p.get_target_ratio(total_days)
            if ratio == 0.5:
                return p.total_shifts * 2  # Double for comparison
            return p.total_shifts

        def get_normalized_day(p: Person) -> float:
            ratio = p.get_target_ratio(total_days)
            if ratio == 0.5:
                return p.day_shifts * 2
            return p.day_shifts

        def get_normalized_night(p: Person) -> float:
            ratio = p.get_target_ratio(total_days)
            if ratio == 0.5:
                return p.night_shifts * 2
            return p.night_shifts

        # Calculate normalized min/max for fairness
        normalized_totals = [get_normalized_shifts(p) for p in self.staff.values()]
        normalized_days = [get_normalized_day(p) for p in self.staff.values()]
        normalized_nights = [get_normalized_night(p) for p in self.staff.values()]

        min_total_norm = min(normalized_totals)
        max_total_norm = max(normalized_totals)
        min_day_norm = min(normalized_days)
        max_day_norm = max(normalized_days)
        min_night_norm = min(normalized_nights)
        max_night_norm = max(normalized_nights)

        person_total_norm = get_normalized_shifts(person)
        person_day_norm = get_normalized_day(person)
        person_night_norm = get_normalized_night(person)

        # STRONG fairness: prioritize people with fewer normalized shifts
        # Total shifts fairness
        if person_total_norm <= min_total_norm:
            score += 50
        elif person_total_norm >= max_total_norm and max_total_norm - min_total_norm >= 1:
            score -= 50

        # Day shift fairness - only compare among night-capable staff
        # Non-night workers naturally need more day shifts, so don't penalize them
        if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
            # Give strong bonus to non-night workers for day shifts
            # They can ONLY do day shifts, so they should be prioritized
            if not person.can_do_night:
                score += 80  # Strong preference for non-night workers
            else:
                # For night-capable staff, apply day fairness among themselves
                night_capable = [p for p in self.staff.values() if p.can_do_night]
                if night_capable:
                    night_cap_days = [get_normalized_day(p) for p in night_capable]
                    min_day_nc = min(night_cap_days)
                    max_day_nc = max(night_cap_days)
                    if person_day_norm <= min_day_nc:
                        score += 30
                    elif person_day_norm >= max_day_nc and max_day_nc - min_day_nc >= 1:
                        score -= 30

        # Night shift fairness
        if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
            if person_night_norm <= min_night_norm:
                score += 30
            elif person_night_norm >= max_night_norm and max_night_norm - min_night_norm >= 1:
                score -= 30

            # Gap bonus - prefer larger gaps
            if person.last_night_shift:
                gap = (d - person.last_night_shift).days
                if gap >= self.MIN_NIGHT_TO_NIGHT_GAP:  # >= 5 days
                    score += 5
                elif gap >= self.EMERGENCY_NIGHT_TO_NIGHT_GAP:  # >= 3 days (emergency)
                    score += 2

        # Weekend fairness - strict control to keep difference <= 1
        # Only compare among non-holiday workers (holiday workers get fewer weekends as compensation)
        if self.is_weekend(d):
            # FIRST: Check holiday penalty (highest priority)
            # 节假日值班的人不应该再排周末
            if person.holiday_shifts > 0:
                # Check if there are others without holiday shifts who can work
                others_no_holiday = any(
                    p.holiday_shifts == 0 and p.name != person.name
                    and self._basic_can_assign(p, d, shift_type)
                    for p in self.staff.values()
                )
                if others_no_holiday:
                    score -= 1000  # Extremely strong penalty - almost a hard block

            # Weekend fairness only among non-holiday workers
            comparable_staff = [p for p in self.staff.values() if p.holiday_shifts == 0]
            if comparable_staff and person.holiday_shifts == 0:
                min_weekend = min(p.weekend_shifts for p in comparable_staff)
                max_weekend = max(p.weekend_shifts for p in comparable_staff)

                # Weekend fairness (secondary to holiday constraint)
                if person.weekend_shifts <= min_weekend:
                    score += 100
                elif person.weekend_shifts > min_weekend:
                    score -= 100

                # Extra penalty if assigning would create difference > 1
                if person.weekend_shifts >= min_weekend + 1:
                    score -= 200

        # 24h shift fairness - 尽量每人最多一次24小时班
        if shift_type == ShiftType.FULL_24H:
            min_24h = min(p.shifts_24h for p in self.staff.values() if p.can_do_24h)

            # Strong penalty if person already has 24h shifts and others don't
            if person.shifts_24h > min_24h:
                score -= 300  # Strong penalty to spread 24h shifts

            # Extra penalty for multiple 24h shifts
            if person.shifts_24h >= 1:
                score -= 200  # Discourage giving same person multiple 24h

        # Thursday Night compensation rule
        # 夜班负担重的人给周四夜班，然后周末双休
        if shift_type == ShiftType.NIGHT and d.weekday() == 3:  # Thursday
            # Bonus for people with MORE night shifts (heavy night burden)
            if person_night_norm >= max_night_norm - 1:
                score += 50  # Strong preference for heavy night workers
            # Also bonus for people with weekend shifts (they deserve compensation)
            if person.weekend_shifts > 0:
                score += 30

        # If person has Thursday night this week, they should NOT work the following weekend
        if self.is_weekend(d):
            # Check if person has Thursday night shift this week
            if d.weekday() == 5:  # Saturday
                thursday_of_this_week = d - timedelta(days=2)
            elif d.weekday() == 6:  # Sunday
                thursday_of_this_week = d - timedelta(days=3)
            else:
                thursday_of_this_week = None

            if thursday_of_this_week and thursday_of_this_week in person.assigned_dates:
                thursday_shift = self.schedule.get((person.name, thursday_of_this_week))
                if thursday_shift in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    score -= 500  # Very strong penalty - Thursday night = weekend off

        # Randomization factor to break ties
        score += random.uniform(0, 0.5)

        return score

    def _assign_shift(self, person: Person, d: date, shift_type: ShiftType):
        """Assign a shift to a person and update their stats."""
        self.schedule[(person.name, d)] = shift_type
        person.assigned_dates.add(d)

        weight = SHIFT_WEIGHTS[shift_type]
        person.weighted_total += weight
        person.total_shifts += 1

        if shift_type == ShiftType.DAY:
            person.day_shifts += 1
            person.last_day_shift = d
        elif shift_type == ShiftType.NIGHT:
            person.night_shifts += 1
            person.last_night_shift = d
        elif shift_type == ShiftType.FULL_24H:
            person.shifts_24h += 1
            person.day_shifts += 1
            person.night_shifts += 1
            person.last_day_shift = d
            person.last_night_shift = d

        if self.is_weekend(d):
            person.weekend_shifts += 1
        if self.is_holiday(d):
            person.holiday_shifts += 1

    def _emergency_can_assign(self, person: Person, d: date, shift_type: ShiftType) -> bool:
        """
        Emergency assignment check - uses relaxed gap rules (3 days for night-to-night).
        Used as fallback when normal 5-day gap cannot be satisfied.
        Checks ALL assigned shifts, not just last_night_shift, for robustness.
        """
        # Check availability period
        if not person.is_available_on(d):
            return False

        # Check fixed off dates
        if d in person.fixed_off_dates:
            return False

        # Check if already assigned this date
        if d in person.assigned_dates:
            return False

        # Check capability
        if shift_type == ShiftType.NIGHT and not person.can_do_night:
            return False
        if shift_type == ShiftType.FULL_24H and not person.can_do_24h:
            return False

        # Emergency gap rules - check against ALL assigned shifts, not just last
        # Use emergency gap (3 days) for night-to-night instead of normal (5 days)
        if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
            # Check gap with ALL night shifts
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    gap = abs((d - assigned_date).days)
                    if gap > 0 and gap < self.EMERGENCY_NIGHT_TO_NIGHT_GAP:
                        return False
            # Also check: if assigning a night shift, are there day shifts within
            # the next MIN_NIGHT_TO_DAY_GAP days that would be violated?
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.DAY, ShiftType.FULL_24H):
                    gap = (assigned_date - d).days  # day_date - night_date
                    if gap > 0 and gap < self.MIN_NIGHT_TO_DAY_GAP:
                        return False  # Day shift too soon after this night shift

        # Day shifts (DAY or 24H) also need gap checks
        if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
            # Night-to-Day gap is a hard constraint (no day shift within 3 days after night)
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    gap = (d - assigned_date).days
                    if gap > 0 and gap < self.MIN_NIGHT_TO_DAY_GAP:
                        return False
            # Day-to-Day gap is also enforced
            for (name, assigned_date), stype in self.schedule.items():
                if name == person.name and stype in (ShiftType.DAY, ShiftType.FULL_24H):
                    gap = abs((d - assigned_date).days)
                    if gap > 0 and gap < self.MIN_DAY_TO_DAY_GAP:
                        return False

        # Weekend constraints (these are hard constraints, not relaxable)
        if self.is_weekend(d):
            # Max 1 shift per weekend per person
            other_day = self._get_weekend_pair(d)
            if other_day and other_day in person.assigned_dates:
                return False

            # Thursday night compensation: if person has Thursday night, skip this weekend
            if self._has_thursday_night_this_week(person, d):
                return False

        return True

    def _select_best_candidate(
        self, d: date, shift_type: ShiftType, exclude: Set[str] = None
    ) -> Optional[Person]:
        """Select the best candidate for a shift based on constraints and scoring."""
        exclude = exclude or set()
        candidates = []

        # First try: normal constraints
        for name, person in self.staff.items():
            if name in exclude:
                continue

            can_assign, reason = self._can_assign_shift(person, d, shift_type)
            if can_assign:
                score = self._calculate_assignment_score(person, d, shift_type)
                candidates.append((person, score))

        # Second try: emergency - ignore gap constraints to ensure coverage
        if not candidates:
            for name, person in self.staff.items():
                if name in exclude:
                    continue

                if self._emergency_can_assign(person, d, shift_type):
                    # Apply heavy penalty for breaking gap rules
                    score = self._calculate_assignment_score(person, d, shift_type) - 500
                    candidates.append((person, score))

        if not candidates:
            return None

        # Sort by score descending and pick the best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _process_fixed_assignments(self):
        """Process all fixed-on assignments first."""
        for person in self.staff.values():
            for d, shift_type in person.fixed_on_dates.items():
                if d in self.dates:
                    self._assign_shift(person, d, shift_type)

    def generate_schedule(self, num_attempts: int = 20) -> bool:
        """
        Generate the monthly schedule by trying multiple times and keeping the best.
        Returns True if successful, False if coverage requirements cannot be met.
        """
        best_schedule = None
        best_stats = None
        best_fairness_score = float('inf')

        for attempt in range(num_attempts):
            # Reset all stats
            for person in self.staff.values():
                person.reset_monthly_stats()

            self.schedule = {}

            # Process fixed assignments first
            self._process_fixed_assignments()

            # Generate one schedule attempt
            self._generate_single_schedule()

            # Validate coverage
            if not self._validate_coverage():
                continue

            # Post-generation optimization: try to balance shifts
            self._optimize_schedule()

            # Calculate fairness score (lower is better)
            score = self._calculate_fairness_score()

            if score < best_fairness_score:
                best_fairness_score = score
                best_schedule = dict(self.schedule)
                best_stats = {name: (
                    p.total_shifts, p.day_shifts, p.night_shifts,
                    p.shifts_24h, p.weekend_shifts, p.holiday_shifts,
                    p.weighted_total, p.last_day_shift, p.last_night_shift,
                    set(p.assigned_dates)
                ) for name, p in self.staff.items()}

        # Restore the best schedule
        if best_schedule:
            self.schedule = best_schedule
            for name, stats in best_stats.items():
                p = self.staff[name]
                (p.total_shifts, p.day_shifts, p.night_shifts,
                 p.shifts_24h, p.weekend_shifts, p.holiday_shifts,
                 p.weighted_total, p.last_day_shift, p.last_night_shift,
                 p.assigned_dates) = stats
            return True

        return False

    def _optimize_schedule(self):
        """
        Post-generation optimization: try to balance weekend, night, and total shifts
        by swapping assignments between people.
        """
        max_iterations = 50  # Prevent infinite loops
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try to balance weekend shifts
            if self._try_balance_weekend_shifts():
                improved = True

            # Try to balance night shifts
            if self._try_balance_night_shifts():
                improved = True

            # Try to reduce burden on high-total people
            if self._try_balance_high_total_burden():
                improved = True

            # Try to balance total shifts (ensure difference <= 1)
            if self._try_balance_total_shifts():
                improved = True

    def _try_balance_weekend_shifts(self) -> bool:
        """Try to swap weekend shifts from high to low count people."""
        weekend_counts = [(name, p.weekend_shifts) for name, p in self.staff.items()]
        if not weekend_counts:
            return False

        max_weekend = max(c[1] for c in weekend_counts)
        min_weekend = min(c[1] for c in weekend_counts)

        # If difference <= 1, no need to balance
        if max_weekend - min_weekend <= 1:
            return False

        # Find people with max and min weekend shifts
        high_people = [name for name, count in weekend_counts if count == max_weekend]
        low_people = [name for name, count in weekend_counts if count == min_weekend]

        # Try to find a swap
        for high_name in high_people:
            high_person = self.staff[high_name]

            # Find weekend shifts assigned to this person
            weekend_shifts = [
                (d, shift_type) for (name, d), shift_type in self.schedule.items()
                if name == high_name and self.is_weekend(d)
            ]

            for d, shift_type in weekend_shifts:
                # Try to find someone with fewer weekend shifts who can take this
                for low_name in low_people:
                    low_person = self.staff[low_name]

                    # Check if low_person can take this shift
                    if self._can_swap_shift(high_person, low_person, d, shift_type):
                        # Perform the swap
                        self._perform_swap(high_person, low_person, d, shift_type)
                        return True

        return False

    def _try_balance_night_shifts(self) -> bool:
        """Try to swap night shifts from high to low count people."""
        total_days = len(self.dates)

        # Only consider night-capable staff
        night_capable = [(name, p) for name, p in self.staff.items() if p.can_do_night]
        if not night_capable:
            return False

        # Normalize for target ratio
        def get_norm_night(p):
            ratio = p.get_target_ratio(total_days)
            return p.night_shifts * 2 if ratio == 0.5 else p.night_shifts

        night_counts = [(name, get_norm_night(p)) for name, p in night_capable]
        max_night = max(c[1] for c in night_counts)
        min_night = min(c[1] for c in night_counts)

        # If difference <= 1, no need to balance
        if max_night - min_night <= 1:
            return False

        # Find people with max and min night shifts
        high_people = [name for name, count in night_counts if count == max_night]
        low_people = [name for name, count in night_counts if count == min_night]

        # Try to find a swap
        for high_name in high_people:
            high_person = self.staff[high_name]

            # Find night shifts assigned to this person
            night_shifts = [
                (d, shift_type) for (name, d), shift_type in self.schedule.items()
                if name == high_name and shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H)
            ]

            for d, shift_type in night_shifts:
                # Skip weekends for 24h swaps (too complex)
                if shift_type == ShiftType.FULL_24H:
                    continue

                # Try to find someone with fewer night shifts who can take this
                for low_name in low_people:
                    low_person = self.staff[low_name]

                    if self._can_swap_shift(high_person, low_person, d, shift_type):
                        self._perform_swap(high_person, low_person, d, shift_type)
                        return True

        return False

    def _try_balance_high_total_burden(self) -> bool:
        """
        Try to reduce night/weekend burden for people with the most total shifts.
        If someone has max total shifts AND max nights/weekends, try to swap some away.
        """
        total_days = len(self.dates)

        # Normalize total shifts for target ratio comparison
        def get_norm_total(p):
            ratio = p.get_target_ratio(total_days)
            return p.total_shifts * 2 if ratio == 0.5 else p.total_shifts

        # Find person(s) with max normalized total shifts
        norm_totals = [(name, get_norm_total(p)) for name, p in self.staff.items()]
        max_total = max(c[1] for c in norm_totals)
        high_total_people = [name for name, count in norm_totals if count == max_total]

        # For each high-total person, check if they also have max nights or weekends
        for high_name in high_total_people:
            high_person = self.staff[high_name]

            # Check weekend burden (only among non-holiday workers)
            non_holiday = [p for p in self.staff.values() if p.holiday_shifts == 0]
            if non_holiday and high_person.holiday_shifts == 0:
                max_weekend = max(p.weekend_shifts for p in non_holiday)
                if high_person.weekend_shifts == max_weekend and max_weekend > 0:
                    # This person has max total AND max weekend - try to swap a weekend shift
                    weekend_shifts = [
                        (d, shift_type) for (name, d), shift_type in self.schedule.items()
                        if name == high_name and self.is_weekend(d)
                    ]
                    for d, shift_type in weekend_shifts:
                        # Find someone with fewer weekends AND fewer total shifts
                        for other_name, other_person in self.staff.items():
                            if other_name == high_name:
                                continue
                            if other_person.holiday_shifts > 0:
                                continue  # Don't swap to holiday workers
                            if other_person.weekend_shifts >= high_person.weekend_shifts:
                                continue  # Must have fewer weekends
                            if get_norm_total(other_person) >= max_total:
                                continue  # Must have fewer total shifts
                            if self._can_swap_shift(high_person, other_person, d, shift_type):
                                self._perform_swap(high_person, other_person, d, shift_type)
                                return True

            # Check night burden (only among night-capable)
            if high_person.can_do_night:
                night_capable = [p for p in self.staff.values() if p.can_do_night]
                if night_capable:
                    def get_norm_night(p):
                        ratio = p.get_target_ratio(total_days)
                        return p.night_shifts * 2 if ratio == 0.5 else p.night_shifts

                    max_night = max(get_norm_night(p) for p in night_capable)
                    if get_norm_night(high_person) == max_night and high_person.night_shifts > 0:
                        # This person has max total AND max nights - try to swap a night shift
                        night_shifts = [
                            (d, shift_type) for (name, d), shift_type in self.schedule.items()
                            if name == high_name and shift_type == ShiftType.NIGHT
                        ]
                        for d, shift_type in night_shifts:
                            # Skip weekend nights (handled above)
                            if self.is_weekend(d):
                                continue
                            # Find someone with fewer nights AND fewer total shifts
                            for other_name, other_person in self.staff.items():
                                if other_name == high_name:
                                    continue
                                if not other_person.can_do_night:
                                    continue
                                if get_norm_night(other_person) >= get_norm_night(high_person):
                                    continue  # Must have fewer nights
                                if get_norm_total(other_person) >= max_total:
                                    continue  # Must have fewer total shifts
                                if self._can_swap_shift(high_person, other_person, d, shift_type):
                                    self._perform_swap(high_person, other_person, d, shift_type)
                                    return True

        return False

    def _try_balance_total_shifts(self) -> bool:
        """
        Try to balance total shifts so that the difference is at most 1.
        Uses normalized shifts (accounting for target ratio - half-month people count as 2x).
        """
        total_days = len(self.dates)

        # Normalize total shifts for target ratio comparison
        def get_norm_total(p):
            ratio = p.get_target_ratio(total_days)
            return p.total_shifts * 2 if ratio == 0.5 else p.total_shifts

        norm_totals = [(name, get_norm_total(p)) for name, p in self.staff.items()]
        if not norm_totals:
            return False

        max_total = max(c[1] for c in norm_totals)
        min_total = min(c[1] for c in norm_totals)

        # If difference <= 1, no need to balance
        if max_total - min_total <= 1:
            return False

        # Find people with max and min total shifts
        high_people = [name for name, count in norm_totals if count == max_total]
        low_people = [name for name, count in norm_totals if count == min_total]

        # Try to find a swap: move a shift from high to low
        for high_name in high_people:
            high_person = self.staff[high_name]

            # Get all shifts for this person, sorted by date
            all_shifts = [
                (d, shift_type) for (name, d), shift_type in self.schedule.items()
                if name == high_name
            ]
            all_shifts.sort(key=lambda x: x[0])

            for d, shift_type in all_shifts:
                # Try to find someone with fewer total shifts who can take this
                for low_name in low_people:
                    low_person = self.staff[low_name]

                    # Check if low_person can take this shift
                    if self._can_swap_shift(high_person, low_person, d, shift_type):
                        # Verify the swap would improve balance
                        new_high_norm = get_norm_total(high_person) - (2 if high_person.get_target_ratio(total_days) == 0.5 else 1)
                        new_low_norm = get_norm_total(low_person) + (2 if low_person.get_target_ratio(total_days) == 0.5 else 1)

                        # Only swap if it improves or maintains balance
                        if new_high_norm >= new_low_norm - 1:
                            self._perform_swap(high_person, low_person, d, shift_type)
                            return True

        return False

    def _get_all_night_shifts(self, person: Person) -> List[date]:
        """Get all night shift dates for a person from the schedule."""
        night_dates = []
        for (name, assigned_date), shift_type in self.schedule.items():
            if name == person.name and shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                night_dates.append(assigned_date)
        return sorted(night_dates)

    def _get_all_day_shifts(self, person: Person) -> List[date]:
        """Get all day shift dates for a person from the schedule."""
        day_dates = []
        for (name, assigned_date), shift_type in self.schedule.items():
            if name == person.name and shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
                day_dates.append(assigned_date)
        return sorted(day_dates)

    def _check_night_gap_with_all_shifts(self, person: Person, d: date, min_gap: int) -> bool:
        """Check if assigning a night shift on date d violates gap constraints with ALL existing night shifts."""
        night_shifts = self._get_all_night_shifts(person)
        for existing_date in night_shifts:
            gap = abs((d - existing_date).days)
            if gap > 0 and gap < min_gap:
                return False  # Violation
        return True  # No violation

    def _check_day_gap_with_all_shifts(self, person: Person, d: date, min_gap: int) -> bool:
        """Check if assigning a day shift on date d violates gap constraints with ALL existing day shifts."""
        day_shifts = self._get_all_day_shifts(person)
        for existing_date in day_shifts:
            gap = abs((d - existing_date).days)
            if gap > 0 and gap < min_gap:
                return False  # Violation
        return True  # No violation

    def _check_night_to_day_gap(self, person: Person, d: date, min_gap: int) -> bool:
        """Check if assigning a day shift on date d violates night-to-day gap constraints."""
        night_shifts = self._get_all_night_shifts(person)
        for night_date in night_shifts:
            gap = (d - night_date).days
            # Only check forward: day shift should not be within min_gap days AFTER a night shift
            if gap > 0 and gap < min_gap:
                return False  # Violation
        return True  # No violation

    def _check_reverse_night_to_day_gap(self, person: Person, night_date: date, min_gap: int) -> bool:
        """Check if assigning a night shift would violate gaps with EXISTING day shifts.

        When assigning a night shift, we need to ensure there are no day shifts
        within the next min_gap days (since day shifts should not come too soon
        after a night shift).
        """
        day_shifts = self._get_all_day_shifts(person)
        for day_date in day_shifts:
            gap = (day_date - night_date).days  # positive if day is after night
            if gap > 0 and gap < min_gap:
                return False  # Existing day shift is too soon after this night shift
        return True  # No violation

    def _can_swap_shift(self, from_person: Person, to_person: Person, d: date, shift_type: ShiftType) -> bool:
        """Check if a shift can be swapped from one person to another."""
        # Check basic capability
        if shift_type == ShiftType.NIGHT and not to_person.can_do_night:
            return False
        if shift_type == ShiftType.FULL_24H and not to_person.can_do_24h:
            return False

        # Check if to_person is available on this date
        if not to_person.is_available_on(d):
            return False
        if d in to_person.fixed_off_dates:
            return False
        if d in to_person.assigned_dates:
            return False

        # Check gap constraints for to_person - check against ALL shifts, not just last
        if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
            if not self._check_night_gap_with_all_shifts(to_person, d, self.MIN_NIGHT_TO_NIGHT_GAP):
                return False
            # Also check reverse: existing day shifts that would be too close after this night
            if not self._check_reverse_night_to_day_gap(to_person, d, self.MIN_NIGHT_TO_DAY_GAP):
                return False

        if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
            # Night-to-Day gap check
            if not self._check_night_to_day_gap(to_person, d, self.MIN_NIGHT_TO_DAY_GAP):
                return False
            # Day-to-Day gap check
            if not self._check_day_gap_with_all_shifts(to_person, d, self.MIN_DAY_TO_DAY_GAP):
                return False

        # Don't swap to holiday workers for weekend shifts
        if self.is_weekend(d) and to_person.holiday_shifts > 0:
            return False

        # Weekend constraints
        if self.is_weekend(d):
            # Max 1 shift per weekend per person
            other_day = self._get_weekend_pair(d)
            if other_day and other_day in to_person.assigned_dates:
                return False

            # Thursday night compensation: if person has Thursday night, skip this weekend
            if self._has_thursday_night_this_week(to_person, d):
                return False

        # Thursday night constraint (reverse check): if swapping Thursday night TO someone,
        # check if they already have weekend shifts for that week
        if d.weekday() == 3 and shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
            sat = d + timedelta(days=2)
            sun = d + timedelta(days=3)
            if sat in to_person.assigned_dates or sun in to_person.assigned_dates:
                return False

        return True

    def _perform_swap(self, from_person: Person, to_person: Person, d: date, shift_type: ShiftType):
        """Perform the actual swap of a shift from one person to another."""
        # Remove from from_person
        del self.schedule[(from_person.name, d)]
        from_person.assigned_dates.discard(d)
        from_person.total_shifts -= 1
        from_person.weighted_total -= SHIFT_WEIGHTS[shift_type]

        if shift_type == ShiftType.DAY:
            from_person.day_shifts -= 1
        elif shift_type == ShiftType.NIGHT:
            from_person.night_shifts -= 1
        elif shift_type == ShiftType.FULL_24H:
            from_person.shifts_24h -= 1
            from_person.day_shifts -= 1
            from_person.night_shifts -= 1

        if self.is_weekend(d):
            from_person.weekend_shifts -= 1
        if self.is_holiday(d):
            from_person.holiday_shifts -= 1

        # Add to to_person
        self._assign_shift(to_person, d, shift_type)

    def _calculate_fairness_score(self) -> float:
        """
        Calculate a fairness score for the current schedule.
        Lower score = more fair. Considers weekend, night, day, and total shift variance.

        Special cases (allowed to have diff > 1):
        - Weekend: only compare among non-holiday workers (holiday workers get fewer weekends)
        - Day shifts: only compare among night-capable staff (non-night workers get more days)
        - Night shifts: only compare among night-capable staff
        """
        import statistics

        total_days = len(self.dates)

        # Get normalized values for comparison (accounting for target ratios)
        def normalize(p, value):
            ratio = p.get_target_ratio(total_days)
            return value * 2 if ratio == 0.5 else value

        # Filter staff by capability/situation for fair comparison
        night_capable = [p for p in self.staff.values() if p.can_do_night]
        non_holiday_workers = [p for p in self.staff.values() if p.holiday_shifts == 0]

        # Weekend: only compare among non-holiday workers
        weekend_shifts = [p.weekend_shifts for p in non_holiday_workers] if non_holiday_workers else []

        # Night: only compare among night-capable staff
        night_shifts = [normalize(p, p.night_shifts) for p in night_capable] if night_capable else []

        # Day: only compare among night-capable staff (non-night workers naturally have more)
        day_shifts = [normalize(p, p.day_shifts) for p in night_capable] if night_capable else []

        # Total: compare all staff (normalized for target ratio)
        total_shifts = [normalize(p, p.total_shifts) for p in self.staff.values()]

        def calc_range(lst):
            return max(lst) - min(lst) if lst else 0

        def calc_variance(lst):
            return statistics.variance(lst) if len(lst) > 1 else 0

        # Score components (weighted)
        # Weekend and Night fairness are most important
        weekend_range = calc_range(weekend_shifts)
        night_range = calc_range(night_shifts)
        day_range = calc_range(day_shifts)
        total_range = calc_range(total_shifts)

        # Heavy penalty if range > 1 (only for comparable groups)
        score = 0
        score += weekend_range * 100 + (50 if weekend_range > 1 else 0)
        score += night_range * 80 + (40 if night_range > 1 else 0)
        score += day_range * 60 + (30 if day_range > 1 else 0)
        score += total_range * 40

        # Add variance for tie-breaking
        score += calc_variance(weekend_shifts) * 10 if weekend_shifts else 0
        score += calc_variance(night_shifts) * 8 if night_shifts else 0
        score += calc_variance(day_shifts) * 6 if day_shifts else 0

        # Heavy penalty if holiday workers have weekend shifts
        for p in self.staff.values():
            if p.holiday_shifts > 0 and p.weekend_shifts > 0:
                score += 200  # Strong penalty

        # Penalty for multiple 24h shifts per person
        for p in self.staff.values():
            if p.shifts_24h > 1:
                score += 100 * (p.shifts_24h - 1)

        return score

    def _generate_single_schedule(self):
        """Generate a single schedule attempt."""
        # Process each day (skip holidays - they are handled via FixedOn lottery)
        for d in self.dates:
            # Skip holidays - holiday shifts are assigned via FixedOn (lottery)
            if self.is_holiday(d):
                continue

            # Determine required shifts for this day
            day_slots_needed = self.day_shifts_per_day
            night_slots_needed = self.night_shifts_per_day

            # Check how many are already filled by fixed assignments
            assigned_today = set()
            for (name, assigned_date), shift_type in self.schedule.items():
                if assigned_date == d:
                    assigned_today.add(name)
                    if shift_type == ShiftType.DAY:
                        day_slots_needed -= 1
                    elif shift_type == ShiftType.NIGHT:
                        night_slots_needed -= 1
                    elif shift_type == ShiftType.FULL_24H:
                        day_slots_needed -= 1
                        night_slots_needed -= 1

            # Fill Day shifts
            for _ in range(max(0, day_slots_needed)):
                candidate = self._select_best_candidate(
                    d, ShiftType.DAY, exclude=assigned_today
                )
                if candidate:
                    # Check if we should consider 24h shift
                    # IMPORTANT: Do NOT assign 24h shifts on weekends (unfair double burden)
                    consider_24h = (
                        night_slots_needed > 0
                        and not self.is_weekend(d)  # No 24h on weekends
                        and random.random() < 0.15  # 15% chance to consider 24h
                    )

                    if consider_24h:
                        # Find best candidate specifically for 24h (considers 24h fairness)
                        best_24h = self._select_best_candidate(
                            d, ShiftType.FULL_24H, exclude=assigned_today
                        )
                        if best_24h and best_24h.shifts_24h == 0:
                            # Only assign 24h if this person hasn't had one yet
                            self._assign_shift(best_24h, d, ShiftType.FULL_24H)
                            assigned_today.add(best_24h.name)
                            night_slots_needed -= 1
                            continue

                    self._assign_shift(candidate, d, ShiftType.DAY)
                    assigned_today.add(candidate.name)

            # Fill Night shifts
            for _ in range(max(0, night_slots_needed)):
                candidate = self._select_best_candidate(
                    d, ShiftType.NIGHT, exclude=assigned_today
                )
                if candidate:
                    self._assign_shift(candidate, d, ShiftType.NIGHT)
                    assigned_today.add(candidate.name)

        return self._validate_coverage()

    def _validate_coverage(self) -> bool:
        """Validate that all non-holiday days have required coverage."""
        for d in self.dates:
            # Skip holidays - coverage is handled via FixedOn (lottery)
            if self.is_holiday(d):
                continue

            day_count = 0
            night_count = 0

            for (name, assigned_date), shift_type in self.schedule.items():
                if assigned_date == d:
                    if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
                        day_count += 1
                    if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                        night_count += 1

            if day_count < 1 or night_count < 1:
                return False

        return True

    def get_schedule_dict(self) -> Dict[Tuple[str, date], str]:
        """Return schedule as dict with string shift types."""
        return {
            (name, d): shift_type.value
            for (name, d), shift_type in self.schedule.items()
        }

    def get_staff_stats(self) -> Dict[str, dict]:
        """Return statistics for all staff."""
        return {name: person.get_stats_dict() for name, person in self.staff.items()}

    def get_fairness_metrics(self) -> dict:
        """Calculate fairness metrics for the schedule."""
        import statistics

        if not self.staff:
            return {}

        total_shifts = [p.total_shifts for p in self.staff.values()]
        night_shifts = [p.night_shifts for p in self.staff.values()]
        weekend_shifts = [p.weekend_shifts for p in self.staff.values()]
        weighted = [p.weighted_total for p in self.staff.values()]

        def safe_stdev(data):
            return statistics.stdev(data) if len(data) > 1 else 0

        return {
            "total_shifts_mean": statistics.mean(total_shifts),
            "total_shifts_stdev": safe_stdev(total_shifts),
            "night_shifts_mean": statistics.mean(night_shifts),
            "night_shifts_stdev": safe_stdev(night_shifts),
            "weekend_shifts_mean": statistics.mean(weekend_shifts),
            "weekend_shifts_stdev": safe_stdev(weekend_shifts),
            "weighted_mean": statistics.mean(weighted),
            "weighted_stdev": safe_stdev(weighted),
        }

    def reschedule_for_leave(
        self, leave_person_name: str, leave_dates: Set[date]
    ) -> List[dict]:
        """
        Reschedule shifts when someone takes temporary leave.

        Args:
            leave_person_name: Name of person taking leave
            leave_dates: Set of dates they are on leave

        Returns:
            List of changes made: [{"date": d, "shift": type, "from": name, "to": name}, ...]
        """
        changes = []

        if leave_person_name not in self.staff:
            return changes

        leave_person = self.staff[leave_person_name]

        # Find shifts that need to be reassigned
        shifts_to_reassign = []
        for (name, d), shift_type in list(self.schedule.items()):
            if name == leave_person_name and d in leave_dates:
                shifts_to_reassign.append((d, shift_type))

        # Sort by date
        shifts_to_reassign.sort(key=lambda x: x[0])

        # For each shift, find a replacement
        for d, shift_type in shifts_to_reassign:
            # Remove the shift from leave person
            del self.schedule[(leave_person_name, d)]
            leave_person.assigned_dates.discard(d)

            # Update stats for leave person
            weight = SHIFT_WEIGHTS[shift_type]
            leave_person.weighted_total -= weight
            leave_person.total_shifts -= 1
            if shift_type == ShiftType.DAY:
                leave_person.day_shifts -= 1
            elif shift_type == ShiftType.NIGHT:
                leave_person.night_shifts -= 1
            elif shift_type == ShiftType.FULL_24H:
                leave_person.shifts_24h -= 1
                leave_person.day_shifts -= 1
                leave_person.night_shifts -= 1
            if self.is_weekend(d):
                leave_person.weekend_shifts -= 1
            if self.is_holiday(d):
                leave_person.holiday_shifts -= 1

            # Find replacement - prioritize people with fewer total shifts
            # MUST find someone to maintain coverage
            replacement = self._find_replacement(d, shift_type, leave_person_name)

            if replacement:
                self._assign_shift(replacement, d, shift_type)
                changes.append({
                    "date": d,
                    "shift_type": shift_type.value,
                    "from": leave_person_name,
                    "to": replacement.name,
                })
            else:
                # This should NEVER happen - restore the original assignment
                # to maintain coverage
                self._assign_shift(leave_person, d, shift_type)
                changes.append({
                    "date": d,
                    "shift_type": shift_type.value,
                    "from": leave_person_name,
                    "to": leave_person_name,  # Keep original (no replacement available)
                    "warning": "No replacement found - original kept",
                })

        return changes

    def _find_replacement(
        self, d: date, shift_type: ShiftType, exclude_name: str
    ) -> Optional[Person]:
        """
        Find a replacement for a shift, prioritizing people with fewer shifts.
        MUST find someone - coverage is mandatory.
        """
        candidates = []

        def calculate_priority(person: Person) -> float:
            """Calculate priority score - lower total shifts = higher priority."""
            score = -person.total_shifts * 100

            if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                score -= person.night_shifts * 10
            else:
                score -= person.day_shifts * 10

            if self.is_weekend(d):
                score -= person.weekend_shifts * 50

            return score

        # First try: normal constraints (respects gap rules)
        for name, person in self.staff.items():
            if name == exclude_name:
                continue

            can_assign, _ = self._can_assign_shift(person, d, shift_type)
            if can_assign:
                candidates.append((person, calculate_priority(person)))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # Second try: emergency - only check absolute constraints
        for name, person in self.staff.items():
            if name == exclude_name:
                continue

            if self._emergency_can_assign(person, d, shift_type):
                candidates.append((person, calculate_priority(person) - 500))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # Third try: FORCE find someone - but still respect MINIMUM gap constraints
        # Night-to-night must be at least EMERGENCY gap (3 days) - this is an absolute minimum
        # This ensures we maintain safety constraints while still finding coverage
        for name, person in self.staff.items():
            if name == exclude_name:
                continue

            # Only check if they CAN do the shift type
            if shift_type == ShiftType.NIGHT and not person.can_do_night:
                continue
            if shift_type == ShiftType.FULL_24H and not person.can_do_24h:
                continue

            # Skip if already assigned this day (can't do two shifts same day)
            if d in person.assigned_dates:
                continue

            # STILL enforce minimum gap constraints even in FORCE mode
            # These are absolute safety constraints that cannot be violated
            if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                if not self._check_night_gap_with_all_shifts(person, d, self.EMERGENCY_NIGHT_TO_NIGHT_GAP):
                    continue  # Must have at least 3 days between night shifts
                # Also check: existing day shifts that would be too close after this night
                if not self._check_reverse_night_to_day_gap(person, d, self.MIN_NIGHT_TO_DAY_GAP):
                    continue  # No day shift within 3 days after this night shift

            if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
                if not self._check_night_to_day_gap(person, d, self.MIN_NIGHT_TO_DAY_GAP):
                    continue  # No day shift within 3 days after night
                if not self._check_day_gap_with_all_shifts(person, d, self.MIN_DAY_TO_DAY_GAP):
                    continue  # Must have at least 3 days between day shifts

            candidates.append((person, calculate_priority(person) - 1000))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # Fourth try: ABSOLUTE FORCE with minimal constraints
        # Still enforce CRITICAL gap constraints to prevent unsafe scheduling:
        # - Night-to-Night: minimum 3 days (emergency gap)
        # - Night-to-Day: minimum 3 days (no day shift within 3 days after night)
        # - Day-to-Day: minimum 3 days
        # These are non-negotiable safety constraints.
        for name, person in self.staff.items():
            if name == exclude_name:
                continue

            if shift_type == ShiftType.NIGHT and not person.can_do_night:
                continue
            if shift_type == ShiftType.FULL_24H and not person.can_do_24h:
                continue

            if d in person.assigned_dates:
                continue

            # CRITICAL: Always enforce minimum gap constraints
            # Night shifts
            if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                # Check night-to-night gap (emergency minimum: 3 days)
                if not self._check_night_gap_with_all_shifts(person, d, self.EMERGENCY_NIGHT_TO_NIGHT_GAP):
                    continue
                # Check reverse night-to-day: no day shift within 3 days after this night
                if not self._check_reverse_night_to_day_gap(person, d, self.MIN_NIGHT_TO_DAY_GAP):
                    continue

            # Day shifts
            if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
                # Check day-to-day gap (minimum: 3 days)
                if not self._check_day_gap_with_all_shifts(person, d, self.MIN_DAY_TO_DAY_GAP):
                    continue
                # Check night-to-day: no day shift within 3 days after night
                if not self._check_night_to_day_gap(person, d, self.MIN_NIGHT_TO_DAY_GAP):
                    continue

            candidates.append((person, calculate_priority(person) - 2000))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # If truly no one can take the shift without violating critical gaps,
        # return None. The caller will handle this (either by alerting user
        # or keeping the original assignment).
        return None

    def get_coverage_summary(self) -> List[dict]:
        """Get coverage summary for each day."""
        summary = []
        for d in self.dates:
            day_staff = []
            night_staff = []

            for (name, assigned_date), shift_type in self.schedule.items():
                if assigned_date == d:
                    if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
                        day_staff.append(name)
                    if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                        night_staff.append(name)

            summary.append({
                "date": d,
                "day_of_week": d.strftime("%a"),
                "is_weekend": self.is_weekend(d),
                "is_holiday": self.is_holiday(d),
                "day_coverage": len(day_staff),
                "night_coverage": len(night_staff),
                "day_staff": day_staff,
                "night_staff": night_staff,
            })

        return summary


def create_staff_from_dataframe(df, year: int, month: int) -> List[Person]:
    """
    Create Person objects from a pandas DataFrame.

    Expected columns:
    - Name: str
    - CanDoNight: bool
    - CanDo24h: bool
    - FixedOff: str (comma-separated dates or ranges like "17-20")
    - FixedOn: str (comma-separated "date:shift_type")
    """
    from utils import parse_date_list

    staff = []

    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name:
            continue

        # Parse fixed off dates (supports ranges like "17-20")
        fixed_off_str = str(row.get("FixedOff", ""))
        fixed_off = set(parse_date_list(fixed_off_str, year, month))

        # Parse fixed on dates (format: "1:Day,5:Night")
        fixed_on = {}
        fixed_on_str = str(row.get("FixedOn", ""))
        if fixed_on_str and fixed_on_str.strip() and fixed_on_str.lower() != "nan":
            # Normalize separators: Chinese comma/colon to English
            fixed_on_str = fixed_on_str.replace("，", ",").replace("；", ",").replace("：", ":")
            for part in fixed_on_str.split(","):
                part = part.strip()
                if ":" in part:
                    try:
                        date_part, shift_part = part.split(":", 1)
                        date_part = date_part.strip()
                        shift_part = shift_part.strip()

                        # Parse date
                        if "-" in date_part:
                            d = date.fromisoformat(date_part)
                        else:
                            d = date(year, month, int(date_part))

                        # Parse shift type
                        shift_map = {
                            "Day": ShiftType.DAY,
                            "Night": ShiftType.NIGHT,
                            "24h": ShiftType.FULL_24H,
                        }
                        if shift_part in shift_map:
                            fixed_on[d] = shift_map[shift_part]
                    except (ValueError, TypeError):
                        continue

        person = Person(
            name=name,
            can_do_night=bool(row.get("CanDoNight", True)),
            can_do_24h=bool(row.get("CanDo24h", True)),
            fixed_off_dates=fixed_off,
            fixed_on_dates=fixed_on,
        )
        staff.append(person)

    return staff

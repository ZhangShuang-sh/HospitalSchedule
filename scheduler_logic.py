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

    # Gap constraints (values are date differences, so "间隔3天" means difference >= 4)
    # Example: shift on day 27, next allowed on day 31 (27+4=31, with 28,29,30 as 3 gap days)
    MIN_NIGHT_TO_NIGHT_GAP = 6      # Normal: 5 days between night shifts (diff >= 6)
    EMERGENCY_NIGHT_TO_NIGHT_GAP = 4  # Fallback: 3 days between if not enough staff (diff >= 4)
    MIN_DAY_TO_DAY_GAP = 4          # 3 days between day shifts (diff >= 4)
    MIN_NIGHT_TO_DAY_GAP = 4        # After night, 3 days before day shift (diff >= 4)

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

        # Pre-calculate holiday workers (people with fixed_on dates that are holidays)
        # This is a HARD constraint that must be known BEFORE scheduling starts
        self.holiday_workers: Set[str] = set()
        for person in self.staff.values():
            for d in person.fixed_on_dates.keys():
                if d in self.holidays:
                    self.holiday_workers.add(person.name)
                    break

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

    def is_holiday_worker(self, person: Person) -> bool:
        """
        Check if person is a holiday worker (has fixed_on dates that are holidays).
        This is determined at initialization and does NOT change during scheduling.
        Holiday workers CANNOT work weekends - this is a HARD constraint.
        """
        return person.name in self.holiday_workers

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
            # HARD CONSTRAINT: Holiday workers CANNOT work weekends
            if self.is_holiday_worker(person):
                return False

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

        # HARD CONSTRAINT: Holiday workers CANNOT work weekends (strict rule)
        # Use is_holiday_worker (pre-calculated) OR holiday_shifts > 0 (runtime)
        if self.is_weekend(d):
            if self.is_holiday_worker(person) or person.holiday_shifts > 0:
                return False, "Holiday worker cannot work weekend (strict rule)"

        # HARD CONSTRAINT: Weekend fairness - max difference of 1
        # Only compare among non-holiday workers (holiday workers are excluded from weekends)
        if self.is_weekend(d):
            # Weekend fairness only among non-holiday workers
            comparable_staff = [p for p in self.staff.values()
                               if not self.is_holiday_worker(p) and p.holiday_shifts == 0]
            if comparable_staff and not self.is_holiday_worker(person) and person.holiday_shifts == 0:
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
            if self.is_holiday_worker(person) or person.holiday_shifts > 0:
                # Check if there are others without holiday shifts who can work
                others_no_holiday = any(
                    not self.is_holiday_worker(p) and p.holiday_shifts == 0 and p.name != person.name
                    and self._basic_can_assign(p, d, shift_type)
                    for p in self.staff.values()
                )
                if others_no_holiday:
                    score -= 1000  # Extremely strong penalty - almost a hard block

            # Weekend fairness only among non-holiday workers
            comparable_staff = [p for p in self.staff.values()
                               if not self.is_holiday_worker(p) and p.holiday_shifts == 0]
            if comparable_staff and not self.is_holiday_worker(person) and person.holiday_shifts == 0:
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

    def _assign_shift(self, person: Person, d: date, shift_type: ShiftType) -> bool:
        """
        Assign a shift to a person and update their stats.
        Returns True if assignment was successful, False if blocked by hard constraint.
        """
        # HARD CONSTRAINT CHECK: Holiday workers CANNOT be assigned weekend shifts
        if self.is_weekend(d):
            # Check if this person is or will be a holiday worker
            is_holiday_worker = self.is_holiday_worker(person) or person.holiday_shifts > 0
            if is_holiday_worker:
                # BLOCK: Do not assign weekend shift to holiday worker
                return False

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

        return True

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

        # HARD CONSTRAINT: Holiday workers CANNOT work weekends (even in emergency)
        if self.is_weekend(d) and (self.is_holiday_worker(person) or person.holiday_shifts > 0):
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

    def generate_schedule(self, num_attempts: int = 100) -> bool:
        """
        Generate the monthly schedule using Multi-Stage Assignment approach.

        This approach avoids infinite loops by assigning shifts in a specific order:
        Phase 0: Initialize - identify holiday workers and weekend-eligible staff
        Phase 1: Assign ALL weekend shifts FIRST (critical for avoiding conflicts)
        Phase 2: Assign Thursday night shifts (compensation rule)
        Phase 3: Fill remaining weekday slots
        Phase 4: Validate and retry if needed (no swapping loops)

        Returns True if successful, False if coverage requirements cannot be met.
        """
        best_schedule = None
        best_stats = None
        best_fairness_score = float('inf')

        # Track fairness relaxation level (start strict, relax if needed)
        max_weekend_diff = 1  # Start with strict fairness

        for attempt in range(num_attempts):
            # Reset all stats
            for person in self.staff.values():
                person.reset_monthly_stats()

            self.schedule = {}

            # Seed randomization for variety between attempts
            random.seed(attempt * 42 + random.randint(0, 1000))

            # ========== PHASE 0: INITIALIZATION ==========
            # Process fixed assignments first (holidays, etc.)
            self._process_fixed_assignments()

            # Identify weekend-eligible staff (exclude holiday workers)
            weekend_eligible = [
                p for p in self.staff.values()
                if not self.is_holiday_worker(p) and p.holiday_shifts == 0
            ]

            if not weekend_eligible:
                # No one eligible for weekends - this is a problem
                continue

            # ========== PHASE 1: ASSIGN WEEKENDS FIRST ==========
            phase1_success = self._phase1_assign_weekends(
                weekend_eligible, max_weekend_diff
            )
            if not phase1_success:
                # Relax fairness constraint after many failures
                if attempt > 0 and attempt % 20 == 0 and max_weekend_diff < 3:
                    max_weekend_diff += 1
                continue

            # ========== PHASE 2: ASSIGN THURSDAY NIGHTS ==========
            phase2_success = self._phase2_assign_thursday_nights()
            if not phase2_success:
                continue

            # ========== PHASE 3: FILL REMAINING WEEKDAYS ==========
            phase3_success = self._phase3_fill_weekdays()
            if not phase3_success:
                continue

            # ========== PHASE 4: VALIDATION ==========
            if not self._validate_coverage():
                continue

            is_valid, violations = self._validate_hard_constraints()
            if not is_valid:
                continue

            # Calculate fairness score
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

                # If we found a perfect score, stop early
                if score == 0:
                    break

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

    # ==================== MULTI-STAGE SCHEDULING METHODS ====================

    def _phase1_assign_weekends(
        self, weekend_eligible: List[Person], max_diff: int = 1
    ) -> bool:
        """
        PHASE 1: Assign ALL weekend shifts FIRST.

        This is the critical step that prevents conflicts between holiday workers
        and weekend assignments. By assigning weekends first, we ensure:
        1. Holiday workers are never considered for weekend shifts
        2. Weekend fairness is achieved from the start
        3. No complex swapping is needed later

        Args:
            weekend_eligible: List of staff members eligible for weekend shifts
            max_diff: Maximum allowed difference in weekend shifts (default 1)

        Returns:
            True if successful, False if constraints cannot be satisfied
        """
        # Get all weekend dates (exclude holidays that fall on weekends)
        weekend_dates = sorted([d for d in self.weekends if d not in self.holidays])

        if not weekend_dates:
            return True  # No weekends to assign

        # Calculate target weekend shifts per person
        total_weekend_slots = len(weekend_dates) * (self.day_shifts_per_day + self.night_shifts_per_day)
        target_per_person = total_weekend_slots / len(weekend_eligible) if weekend_eligible else 0

        # Group weekends by Sat/Sun pairs
        weekend_pairs = []
        processed = set()
        for d in weekend_dates:
            if d in processed:
                continue
            pair = self._get_weekend_pair(d)
            if pair and pair in weekend_dates:
                weekend_pairs.append((min(d, pair), max(d, pair)))
                processed.add(d)
                processed.add(pair)
            else:
                weekend_pairs.append((d, None))
                processed.add(d)

        # Assign weekend shifts - iterate through each weekend day
        for d in weekend_dates:
            if self.is_holiday(d):
                continue

            # Check if already assigned via fixed assignments
            day_slots = self.day_shifts_per_day
            night_slots = self.night_shifts_per_day

            for (name, assigned_date), shift_type in self.schedule.items():
                if assigned_date == d:
                    if shift_type in (ShiftType.DAY, ShiftType.FULL_24H):
                        day_slots -= 1
                    if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                        night_slots -= 1

            # Assign day shifts
            for _ in range(max(0, day_slots)):
                candidate = self._select_weekend_candidate(
                    d, ShiftType.DAY, weekend_eligible, max_diff
                )
                if candidate:
                    if not self._assign_shift(candidate, d, ShiftType.DAY):
                        return False  # Hard constraint violation
                else:
                    return False  # No valid candidate

            # Assign night shifts
            for _ in range(max(0, night_slots)):
                candidate = self._select_weekend_candidate(
                    d, ShiftType.NIGHT, weekend_eligible, max_diff
                )
                if candidate:
                    if not self._assign_shift(candidate, d, ShiftType.NIGHT):
                        return False
                else:
                    return False

        return True

    def _select_weekend_candidate(
        self, d: date, shift_type: ShiftType, eligible: List[Person], max_diff: int
    ) -> Optional[Person]:
        """
        Select the best candidate for a weekend shift.

        Prioritizes:
        1. Gap constraint satisfaction (CRITICAL)
        2. Weekend fairness (max_diff constraint)
        3. Lower total shifts
        """
        candidates = []

        # Get current min weekend shifts among eligible
        min_weekend = min(p.weekend_shifts for p in eligible) if eligible else 0

        for person in eligible:
            # Skip if already assigned this day
            if d in person.assigned_dates:
                continue

            # Skip if already worked other day of this weekend
            pair = self._get_weekend_pair(d)
            if pair and pair in person.assigned_dates:
                continue

            # CRITICAL: Check gap constraints using _can_assign_shift
            can_assign, _ = self._can_assign_shift(person, d, shift_type)
            if not can_assign:
                # Try emergency rules
                if not self._emergency_can_assign(person, d, shift_type):
                    continue

            # Check fairness constraint
            if person.weekend_shifts > min_weekend + max_diff:
                continue

            # Score: prefer fewer weekend shifts, then fewer total shifts
            score = -person.weekend_shifts * 100 - person.total_shifts
            # Add randomization for tie-breaking
            score += random.uniform(0, 10)
            candidates.append((person, score))

        if not candidates:
            return None

        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _phase2_assign_thursday_nights(self) -> bool:
        """
        PHASE 2: Assign Thursday night shifts.

        Thursday night workers get Fri/Sat/Sun off as compensation.
        Therefore, we must:
        1. Only assign Thursday nights to people NOT working the upcoming weekend
        2. Prioritize people with high night-shift load (give them long weekend)
        3. Still respect all gap constraints

        Returns:
            True if successful, False if constraints cannot be satisfied
        """
        for thursday in self.thursdays:
            if self.is_holiday(thursday):
                continue

            # Check if already assigned
            night_slots = self.night_shifts_per_day
            for (name, assigned_date), shift_type in self.schedule.items():
                if assigned_date == thursday:
                    if shift_type in (ShiftType.NIGHT, ShiftType.FULL_24H):
                        night_slots -= 1

            if night_slots <= 0:
                continue  # Already filled

            # Find candidates - must NOT work upcoming weekend
            sat = thursday + timedelta(days=2)
            sun = thursday + timedelta(days=3)

            for _ in range(night_slots):
                candidate = self._select_thursday_night_candidate(thursday, sat, sun)
                if candidate:
                    if not self._assign_shift(candidate, thursday, ShiftType.NIGHT):
                        # Try next candidate if this fails
                        continue
                else:
                    # No valid candidate - this is not a fatal error for Phase 2
                    # The slot will be filled in Phase 3
                    break

        return True

    def _select_thursday_night_candidate(
        self, thursday: date, sat: date, sun: date
    ) -> Optional[Person]:
        """
        Select the best candidate for Thursday night shift.

        Must NOT work upcoming weekend (compensation rule).
        Prioritizes staff with more night shifts (gives them long weekend benefit).
        """
        candidates = []

        for person in self.staff.values():
            # Skip if cannot do night shifts
            if not person.can_do_night:
                continue

            # Skip if already assigned this day
            if thursday in person.assigned_dates:
                continue

            # CRITICAL: Skip if working upcoming weekend (compensation rule)
            if sat in person.assigned_dates or sun in person.assigned_dates:
                continue

            # Check gap constraints
            can_assign, _ = self._can_assign_shift(person, thursday, ShiftType.NIGHT)
            if not can_assign:
                if not self._emergency_can_assign(person, thursday, ShiftType.NIGHT):
                    continue

            # Score: prioritize those with MORE night shifts (they need the weekend off)
            # Also consider total shifts for balance
            score = person.night_shifts * 10 - person.total_shifts
            score += random.uniform(0, 5)
            candidates.append((person, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _phase3_fill_weekdays(self) -> bool:
        """
        PHASE 3: Fill remaining weekday slots.

        At this point:
        - Weekend shifts are already assigned (Phase 1)
        - Thursday nights are assigned (Phase 2)
        - We just need to fill Mon, Tue, Wed, Fri and remaining slots

        Uses standard greedy scoring with gap constraint enforcement.

        Returns:
            True if successful, False if coverage cannot be achieved
        """
        # Process each non-weekend, non-holiday date
        for d in self.dates:
            # Skip holidays
            if self.is_holiday(d):
                continue

            # Skip weekends (already handled in Phase 1)
            if self.is_weekend(d):
                continue

            # Determine required shifts for this day
            day_slots_needed = self.day_shifts_per_day
            night_slots_needed = self.night_shifts_per_day

            # Check how many are already filled
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
            day_filled = 0
            for _ in range(max(0, day_slots_needed)):
                tried = set()
                assigned = False
                while not assigned:
                    candidate = self._select_best_candidate(
                        d, ShiftType.DAY, exclude=assigned_today | tried
                    )
                    if not candidate:
                        break

                    if self._assign_shift(candidate, d, ShiftType.DAY):
                        assigned_today.add(candidate.name)
                        assigned = True
                        day_filled += 1
                    else:
                        tried.add(candidate.name)

            # Fill Night shifts
            night_filled = 0
            for _ in range(max(0, night_slots_needed)):
                tried = set()
                assigned = False
                while not assigned:
                    candidate = self._select_best_candidate(
                        d, ShiftType.NIGHT, exclude=assigned_today | tried
                    )
                    if not candidate:
                        break

                    if self._assign_shift(candidate, d, ShiftType.NIGHT):
                        assigned_today.add(candidate.name)
                        assigned = True
                        night_filled += 1
                    else:
                        tried.add(candidate.name)

            # Check if this day has sufficient coverage
            if day_filled < day_slots_needed or night_filled < night_slots_needed:
                return False  # Coverage requirement not met

        return True

    def _calculate_fairness_score(self) -> float:
        """
        Calculate a fairness score for the current schedule.
        Lower score = more fair.

        Considers:
        - Weekend shift variance (among non-holiday workers)
        - Night shift variance (among night-capable staff)
        - Total shift variance
        - Hard constraint violations (heavy penalty)
        """
        import statistics

        total_days = len(self.dates)

        def normalize(p, value):
            ratio = p.get_target_ratio(total_days)
            return value * 2 if ratio == 0.5 else value

        # Filter staff by capability
        night_capable = [p for p in self.staff.values() if p.can_do_night]
        non_holiday_workers = [
            p for p in self.staff.values()
            if not self.is_holiday_worker(p) and p.holiday_shifts == 0
        ]

        # Weekend: only compare among non-holiday workers
        weekend_shifts = [p.weekend_shifts for p in non_holiday_workers] if non_holiday_workers else []

        # Night: only compare among night-capable staff
        night_shifts = [normalize(p, p.night_shifts) for p in night_capable] if night_capable else []

        # Total: compare all staff
        total_shifts = [normalize(p, p.total_shifts) for p in self.staff.values()]

        def calc_range(lst):
            return max(lst) - min(lst) if lst else 0

        def calc_variance(lst):
            return statistics.variance(lst) if len(lst) > 1 else 0

        # Calculate score components
        weekend_range = calc_range(weekend_shifts)
        night_range = calc_range(night_shifts)
        total_range = calc_range(total_shifts)

        score = 0
        # Weekend balance is most critical
        score += weekend_range * 200 + (weekend_range ** 2 * 100 if weekend_range > 1 else 0)
        score += night_range * 80 + (40 if night_range > 1 else 0)
        score += total_range * 40

        # Add variance for tie-breaking
        score += calc_variance(weekend_shifts) * 10 if weekend_shifts else 0
        score += calc_variance(night_shifts) * 8 if night_shifts else 0

        # Heavy penalty if holiday workers have weekend shifts (should never happen)
        for p in self.staff.values():
            if (self.is_holiday_worker(p) or p.holiday_shifts > 0) and p.weekend_shifts > 0:
                score += 10000

        return score

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

    def _validate_hard_constraints(self) -> Tuple[bool, List[str]]:
        """
        Validate that all HARD constraints are satisfied.
        Returns (is_valid, list_of_violations).

        Hard constraints that MUST be satisfied:
        1. Holiday workers CANNOT have weekend shifts
        2. Max 1 shift per weekend per person
        3. Thursday night workers should not work that weekend
        """
        violations = []

        # HARD CONSTRAINT 1: Holiday workers CANNOT have weekend shifts
        for person in self.staff.values():
            is_holiday_worker = self.is_holiday_worker(person) or person.holiday_shifts > 0
            if is_holiday_worker and person.weekend_shifts > 0:
                violations.append(
                    f"VIOLATION: {person.name} is a holiday worker but has {person.weekend_shifts} weekend shift(s)"
                )

        # HARD CONSTRAINT 2: Max 1 shift per weekend per person
        for person in self.staff.values():
            # Group weekend shifts by week
            weekend_dates = [
                d for (name, d), shift_type in self.schedule.items()
                if name == person.name and self.is_weekend(d)
            ]
            # Check for same-weekend duplicates (Sat+Sun of same week)
            for d in weekend_dates:
                pair = self._get_weekend_pair(d)
                if pair and pair in weekend_dates:
                    violations.append(
                        f"VIOLATION: {person.name} has shifts on both days of weekend ({d}, {pair})"
                    )
                    break  # Only report once per person

        # HARD CONSTRAINT 3: Thursday night workers should not work that weekend
        for person in self.staff.values():
            for d in self.thursdays:
                shift = self.schedule.get((person.name, d))
                if shift in (ShiftType.NIGHT, ShiftType.FULL_24H):
                    # Check if they work the following weekend
                    sat = d + timedelta(days=2)
                    sun = d + timedelta(days=3)
                    if sat in person.assigned_dates or sun in person.assigned_dates:
                        violations.append(
                            f"VIOLATION: {person.name} has Thursday night ({d}) but also works weekend ({sat}/{sun})"
                        )

        return len(violations) == 0, violations

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
                if self._assign_shift(replacement, d, shift_type):
                    changes.append({
                        "date": d,
                        "shift_type": shift_type.value,
                        "from": leave_person_name,
                        "to": replacement.name,
                    })
                else:
                    # Assignment blocked by hard constraint, restore original
                    self._assign_shift(leave_person, d, shift_type)
                    changes.append({
                        "date": d,
                        "shift_type": shift_type.value,
                        "from": leave_person_name,
                        "to": leave_person_name,
                        "warning": "Replacement blocked by constraint - original kept",
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

            # HARD CONSTRAINT: Holiday workers CANNOT work weekends (even in FORCE mode)
            if self.is_weekend(d) and (self.is_holiday_worker(person) or person.holiday_shifts > 0):
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

            # HARD CONSTRAINT: Holiday workers CANNOT work weekends (even in ABSOLUTE FORCE mode)
            if self.is_weekend(d) and (self.is_holiday_worker(person) or person.holiday_shifts > 0):
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

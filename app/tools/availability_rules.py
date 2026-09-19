"""Deterministic vehicle-availability rules, kept free of any HTTP/Strapi
detail so they can be unit-tested as plain data-in/data-out logic. This
is the one place the PENDING/APPROVED-blocks, DECLINED/CANCELLED/
COMPLETED-don't-block business rule lives - StrapiBusinessServiceClient
calls into it rather than re-deriving it.
"""
from datetime import date

# Only bookings actively holding a reservation should block a requested
# date range. DECLINED/CANCELLED never held the car; COMPLETED already
# gave it back - none of the three say anything about future dates.
BLOCKING_BOOKING_STATUSES = frozenset({"PENDING", "APPROVED"})


def booking_blocks_range(
    *,
    booking_status: str,
    pickup_date: date,
    return_date: date,
    requested_start: date,
    requested_end: date,
) -> bool:
    """True if a single booking conflicts with the requested [start, end]
    range. Strapi's pickupDate/returnDate are whole-day DATE values (no
    time-of-day), so this compares dates directly - no timestamp handling.
    """
    if booking_status not in BLOCKING_BOOKING_STATUSES:
        return False

    return pickup_date <= requested_end and return_date >= requested_start


def car_is_available(
    *,
    car_availability: str,
    bookings: list[dict],
    requested_start: date,
    requested_end: date,
) -> bool:
    """A car is available only if its static fleet flag is AVAILABLE and
    no blocking booking overlaps the requested range. Car.availability is
    never treated as date-specific on its own - see Phase 3.1's audit.

    `bookings` is a list of plain dicts with `bookingStatus`, `pickupDate`,
    `returnDate` (ISO date strings) - the shape Strapi's populate=bookings
    response already provides, so callers don't need to build a separate
    model just to pass data through this check.
    """
    if car_availability != "AVAILABLE":
        return False

    for booking in bookings:
        if booking_blocks_range(
            booking_status=booking["bookingStatus"],
            pickup_date=date.fromisoformat(booking["pickupDate"]),
            return_date=date.fromisoformat(booking["returnDate"]),
            requested_start=requested_start,
            requested_end=requested_end,
        ):
            return False

    return True

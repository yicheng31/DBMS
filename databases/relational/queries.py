"""
TransitFlow — PostgreSQL / Relational Database Layer
=====================================================
This module handles all queries to PostgreSQL.

TWO ROLES ARE SERVED HERE:
  1. Relational  → dual-network transit (metro + national rail),
                   availability, fares, bookings, seat selection
  2. Vector      → policy document similarity search (pgvector)

STUDENT TASK
------------
Design your schema in databases/relational/schema.sql, seed it with
skeleton/seed_postgres.py, then implement the query functions below.

Functions prefixed with `query_`  are read-only lookups called by the agent.
Functions prefixed with `execute_` are write operations (booking/cancellation).

The vector functions (query_policy_vector_search, store_policy_document)
are already implemented — do not modify them.
"""

from __future__ import annotations

import json
import random
import string
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import psycopg2
import psycopg2.extras
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

from skeleton.config import PG_DSN, VECTOR_TOP_K, VECTOR_SIMILARITY_THRESHOLD


PASSWORD_HASHER = PasswordHasher()


def _connect():
    """Return a new psycopg2 connection with autocommit enabled."""
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    return conn


def _gen_booking_id() -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"BK-{suffix}"


def _gen_payment_id() -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"PM-{suffix}"


def _gen_user_id() -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"RU-{suffix}"


# ── Example ───────────────────────────────────────────────────────────────────
# The block below shows the query pattern: open a cursor, run SQL, return rows.
# Use _connect() for read-only queries; for write operations use a manual
# connection with conn.commit() / conn.rollback() (see execute_booking below).

def example_query() -> dict:
    """Example: returns the name of the connected database."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT current_database() AS db;")
            return dict(cur.fetchone())

# Relational QUERY and transaction functions implementation.
# ─────────────────────────────────────────────────────────────────────────────


# ── NATIONAL RAIL AVAILABILITY ────────────────────────────────────────────────

def query_national_rail_availability(
    origin_id: str,
    destination_id: str,
    travel_date: Optional[str] = None,
) -> list[dict]:
    """
    Return national rail schedules that serve both origin and destination stations
    in the correct order, along with seat occupancy for the requested travel date.

    Args:
        origin_id:       e.g. "NR01"
        destination_id:  e.g. "NR05"
        travel_date:     e.g. "2025-06-01" — used to count bookings; omit for general info
    """
    sql = """
        SELECT
            s.schedule_id,
            s.line,
            s.service_type,
            s.direction,
            orig.station_id AS origin_station_id,
            orig.name AS origin_name,
            dest.station_id AS destination_station_id,
            dest.name AS destination_name,
            s.first_train_time::text,
            s.last_train_time::text,
            s.frequency_min,
            origin_stop.stop_sequence AS origin_sequence,
            destination_stop.stop_sequence AS destination_sequence,
            destination_stop.stop_sequence - origin_stop.stop_sequence AS stops_travelled,
            destination_stop.travel_time_from_origin_min
                - origin_stop.travel_time_from_origin_min AS travel_time_min,
            COUNT(seat.id) AS total_seats,
            COUNT(active_booking.id) AS booked_seats,
            COUNT(seat.id) - COUNT(active_booking.id) AS available_seats,
            CASE
                WHEN COUNT(seat.id) = 0 THEN 'not_configured'
                WHEN COUNT(seat.id) - COUNT(active_booking.id) = 0 THEN 'sold_out'
                ELSE 'available'
            END AS availability_status
        FROM national_rail_schedules s
        JOIN national_rail_schedule_stops origin_stop
            ON origin_stop.national_rail_schedule_pk = s.id
        JOIN national_rail_stations orig
            ON orig.id = origin_stop.national_rail_station_pk
        JOIN national_rail_schedule_stops destination_stop
            ON destination_stop.national_rail_schedule_pk = s.id
        JOIN national_rail_stations dest
            ON dest.id = destination_stop.national_rail_station_pk
        LEFT JOIN national_rail_seats seat
            ON seat.national_rail_schedule_pk = s.id
        LEFT JOIN national_rail_bookings active_booking
            ON active_booking.national_rail_schedule_pk = s.id
           AND active_booking.national_rail_seat_pk = seat.id
           AND active_booking.status <> 'cancelled'
           AND (%s::date IS NOT NULL AND active_booking.travel_date = %s::date)
        WHERE orig.station_id = %s
          AND dest.station_id = %s
          AND origin_stop.stop_sequence < destination_stop.stop_sequence
        GROUP BY
            s.id, s.schedule_id, s.line, s.service_type, s.direction,
            orig.station_id, orig.name, dest.station_id, dest.name,
            s.first_train_time, s.last_train_time, s.frequency_min,
            origin_stop.stop_sequence, destination_stop.stop_sequence,
            origin_stop.travel_time_from_origin_min,
            destination_stop.travel_time_from_origin_min
        ORDER BY s.schedule_id
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (travel_date, travel_date, origin_id, destination_id))
            return [dict(row) for row in cur.fetchall()]


def query_national_rail_fare(
    schedule_id: str,
    fare_class: str,
    stops_travelled: int,
) -> Optional[dict]:
    """
    Calculate the fare for a national rail journey.

    Args:
        schedule_id:     e.g. "NR_SCH01"
        fare_class:      "standard" or "first"
        stops_travelled: number of stops between origin and destination (inclusive)

    Returns:
        dict with fare_class, base_fare_usd, per_stop_rate_usd, total_fare_usd
    """
    if stops_travelled <= 0:
        return None

    sql = """
        SELECT
            fare_class,
            base_fare_usd,
            per_stop_rate_usd,
            base_fare_usd + (%s * per_stop_rate_usd) AS total_fare_usd
        FROM national_rail_fares f
        JOIN national_rail_schedules s
            ON s.id = f.national_rail_schedule_pk
        WHERE s.schedule_id = %s
          AND f.fare_class = %s
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (stops_travelled, schedule_id, fare_class))
            row = cur.fetchone()
            return dict(row) if row else None


def query_national_rail_schedule_fares(schedule_id: str) -> list[dict]:
    """
    Return all fare classes for a national rail schedule using the full route.

    This supports questions such as "How much is NR_SCH04?" where the user gives
    only a schedule id and expects the ticket price for that service.
    """
    sql = """
        WITH route_span AS (
            SELECT
                schedule.id AS schedule_pk,
                MIN(stop.stop_sequence) AS first_stop_sequence,
                MAX(stop.stop_sequence) AS last_stop_sequence,
                MAX(stop.stop_sequence) - MIN(stop.stop_sequence) AS stops_travelled,
                MIN(stop.travel_time_from_origin_min) AS first_stop_time_min,
                MAX(stop.travel_time_from_origin_min) AS last_stop_time_min
            FROM national_rail_schedules schedule
            JOIN national_rail_schedule_stops stop
                ON stop.national_rail_schedule_pk = schedule.id
            WHERE schedule.schedule_id = %s
            GROUP BY schedule.id
        )
        SELECT
            schedule.schedule_id,
            schedule.line,
            schedule.service_type,
            schedule.direction,
            origin.station_id AS origin_station_id,
            origin.name AS origin_name,
            destination.station_id AS destination_station_id,
            destination.name AS destination_name,
            schedule.first_train_time::text,
            schedule.last_train_time::text,
            route_span.stops_travelled,
            route_span.last_stop_time_min - route_span.first_stop_time_min AS travel_time_min,
            fare.fare_class,
            fare.base_fare_usd,
            fare.per_stop_rate_usd,
            fare.base_fare_usd + (route_span.stops_travelled * fare.per_stop_rate_usd)
                AS total_fare_usd
        FROM national_rail_schedules schedule
        JOIN route_span
            ON route_span.schedule_pk = schedule.id
        JOIN national_rail_stations origin
            ON origin.id = schedule.origin_station_pk
        JOIN national_rail_stations destination
            ON destination.id = schedule.destination_station_pk
        JOIN national_rail_fares fare
            ON fare.national_rail_schedule_pk = schedule.id
        ORDER BY
            CASE fare.fare_class
                WHEN 'standard' THEN 1
                WHEN 'first' THEN 2
                ELSE 3
            END
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (schedule_id,))
            return [dict(row) for row in cur.fetchall()]


# ── METRO SCHEDULES & FARE ────────────────────────────────────────────────────

def query_metro_schedules(origin_id: str, destination_id: str) -> list[dict]:
    """
    Return metro schedules that serve both origin and destination in the correct order.

    Args:
        origin_id:       e.g. "MS01"
        destination_id:  e.g. "MS09"
    """
    sql = """
        SELECT
            s.schedule_id,
            s.line,
            s.direction,
            orig.station_id AS origin_station_id,
            orig.name AS origin_name,
            dest.station_id AS destination_station_id,
            dest.name AS destination_name,
            s.first_train_time::text,
            s.last_train_time::text,
            s.frequency_min,
            s.base_fare_usd,
            s.per_stop_rate_usd,
            origin_stop.stop_sequence AS origin_sequence,
            destination_stop.stop_sequence AS destination_sequence,
            destination_stop.stop_sequence - origin_stop.stop_sequence AS stops_travelled,
            destination_stop.travel_time_from_origin_min
                - origin_stop.travel_time_from_origin_min AS travel_time_min
        FROM metro_schedules s
        JOIN metro_schedule_stops origin_stop
            ON origin_stop.metro_schedule_pk = s.id
        JOIN metro_stations orig
            ON orig.id = origin_stop.metro_station_pk
        JOIN metro_schedule_stops destination_stop
            ON destination_stop.metro_schedule_pk = s.id
        JOIN metro_stations dest
            ON dest.id = destination_stop.metro_station_pk
        WHERE orig.station_id = %s
          AND dest.station_id = %s
          AND origin_stop.stop_sequence < destination_stop.stop_sequence
        ORDER BY s.schedule_id
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (origin_id, destination_id))
            return [dict(row) for row in cur.fetchall()]


def query_metro_fare(schedule_id: str, stops_travelled: int) -> Optional[dict]:
    """
    Calculate the metro fare for a single-ticket journey.

    Args:
        schedule_id:     e.g. "MS_SCH01"
        stops_travelled: number of stops between origin and destination

    Returns:
        dict with base_fare_usd, per_stop_rate_usd, total_fare_usd
    """
    if stops_travelled <= 0:
        return None

    sql = """
        SELECT
            base_fare_usd,
            per_stop_rate_usd,
            base_fare_usd + (%s * per_stop_rate_usd) AS total_fare_usd
        FROM metro_schedules
        WHERE schedule_id = %s
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (stops_travelled, schedule_id))
            row = cur.fetchone()
            return dict(row) if row else None


def query_metro_fare_estimate(stops_travelled: int) -> Optional[dict]:
    """
    Estimate a metro fare using the configured PostgreSQL metro fare rule.

    This is used when route planning comes from Neo4j and there is no single
    direct metro schedule covering the full journey.
    """
    if stops_travelled <= 0:
        return None

    sql = """
        SELECT
            base_fare_usd,
            per_stop_rate_usd,
            base_fare_usd + (%s * per_stop_rate_usd) AS total_fare_usd
        FROM metro_schedules
        GROUP BY base_fare_usd, per_stop_rate_usd
        ORDER BY COUNT(*) DESC, base_fare_usd, per_stop_rate_usd
        LIMIT 1
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (stops_travelled,))
            row = cur.fetchone()
            return dict(row) if row else None


# ── SEAT SELECTION ────────────────────────────────────────────────────────────

def query_available_seats(
    schedule_id: str,
    travel_date: str,
    fare_class: str,
) -> list[dict]:
    """
    Return available seats for a national rail journey on a given date.

    Args:
        schedule_id:  e.g. "NR_SCH01"
        travel_date:  e.g. "2025-06-01"
        fare_class:   "standard" or "first"

    Returns:
        List of dicts: {seat_id, coach, row, column}
    """
    sql = """
        SELECT
            seat.seat_id,
            seat.coach,
            seat.row_number AS row,
            seat.seat_column AS column
        FROM national_rail_seats seat
        JOIN national_rail_schedules schedule
            ON schedule.id = seat.national_rail_schedule_pk
        WHERE schedule.schedule_id = %s
          AND seat.fare_class = %s
          AND NOT EXISTS (
              SELECT 1
              FROM national_rail_bookings booking
              WHERE booking.national_rail_seat_pk = seat.id
                AND booking.national_rail_schedule_pk = schedule.id
                AND booking.travel_date = %s
                AND booking.status <> 'cancelled'
          )
        ORDER BY seat.coach, seat.row_number, seat.seat_column
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (schedule_id, fare_class, travel_date))
            return [dict(row) for row in cur.fetchall()]


def auto_select_adjacent_seats(available_seats: list[dict], count: int) -> list[str]:
    """
    Select `count` seats that are as close together as possible (same row preferred,
    then adjacent rows). Returns a list of seat_ids.

    Args:
        available_seats: output of query_available_seats()
        count:           number of seats needed
    """
    if not available_seats or count <= 0:
        return []
    if count >= len(available_seats):
        return [s["seat_id"] for s in available_seats[:count]]

    from collections import defaultdict
    rows: dict[int, list[dict]] = defaultdict(list)
    for seat in available_seats:
        rows[seat["row"]].append(seat)

    for row_seats in sorted(rows.values(), key=lambda s: s[0]["row"]):
        if len(row_seats) >= count:
            return [s["seat_id"] for s in row_seats[:count]]

    sorted_seats = sorted(available_seats, key=lambda s: (s["row"], s["column"]))
    return [s["seat_id"] for s in sorted_seats[:count]]


# ── USER & BOOKING QUERIES ────────────────────────────────────────────────────

def query_user_profile(user_email: str) -> Optional[dict]:
    """Return a user's profile by email."""
    sql = """
        SELECT
            user_id,
            email,
            first_name || ' ' || surname AS full_name,
            first_name,
            surname,
            phone,
            date_of_birth,
            registered_at,
            is_active
        FROM registered_users
        WHERE email = %s
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (user_email,))
            row = cur.fetchone()
            return dict(row) if row else None


def query_user_bookings(user_email: str) -> dict:
    """
    Return a user's combined booking history (national rail + metro).

    Returns:
        dict with keys 'national_rail' (list) and 'metro' (list)
    """
    rail_sql = """
        SELECT
            booking.booking_id,
            schedule.schedule_id,
            origin.station_id AS origin_station_id,
            origin.name AS origin_name,
            destination.station_id AS destination_station_id,
            destination.name AS destination_name,
            booking.travel_date,
            booking.departure_time::text,
            booking.ticket_type,
            booking.fare_class,
            seat.coach,
            seat.seat_id,
            booking.stops_travelled,
            booking.fare_usd,
            booking.amount_usd,
            booking.status,
            booking.booked_at,
            booking.travelled_at
        FROM registered_users users
        JOIN national_rail_bookings booking
            ON booking.user_pk = users.id
        JOIN national_rail_schedules schedule
            ON schedule.id = booking.national_rail_schedule_pk
        JOIN national_rail_stations origin
            ON origin.id = booking.origin_station_pk
        JOIN national_rail_stations destination
            ON destination.id = booking.destination_station_pk
        JOIN national_rail_seats seat
            ON seat.id = booking.national_rail_seat_pk
        WHERE users.email = %s
        ORDER BY booking.travel_date DESC, booking.departure_time DESC
    """
    metro_sql = """
        SELECT
            trip.trip_id,
            schedule.schedule_id,
            origin.station_id AS origin_station_id,
            origin.name AS origin_name,
            destination.station_id AS destination_station_id,
            destination.name AS destination_name,
            trip.travel_date,
            trip.ticket_type,
            trip.day_pass_ref,
            trip.stops_travelled,
            trip.fare_usd,
            trip.amount_usd,
            trip.status,
            trip.purchased_at,
            trip.travelled_at
        FROM registered_users users
        JOIN metro_trips trip
            ON trip.user_pk = users.id
        JOIN metro_schedules schedule
            ON schedule.id = trip.metro_schedule_pk
        JOIN metro_stations origin
            ON origin.id = trip.origin_station_pk
        JOIN metro_stations destination
            ON destination.id = trip.destination_station_pk
        WHERE users.email = %s
        ORDER BY trip.travel_date DESC, trip.purchased_at DESC
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(rail_sql, (user_email,))
            national_rail = [dict(row) for row in cur.fetchall()]
            cur.execute(metro_sql, (user_email,))
            metro = [dict(row) for row in cur.fetchall()]
            return {"national_rail": national_rail, "metro": metro}


def query_payment_info(booking_id: str, user_email: str) -> Optional[dict]:
    """Return payment record for a booking or metro trip owned by the user."""
    sql = """
        SELECT
            payment.payment_id,
            COALESCE(booking.booking_id, trip.trip_id) AS booking_id,
            CASE
                WHEN booking.id IS NOT NULL THEN 'national_rail'
                ELSE 'metro'
            END AS network_type,
            payment.amount_usd,
            payment.refunded_amount_usd,
            payment.method,
            payment.status,
            payment.paid_at,
            payment.refunded_at
        FROM payments payment
        LEFT JOIN national_rail_bookings booking
            ON booking.id = payment.national_rail_booking_pk
        LEFT JOIN registered_users booking_user
            ON booking_user.id = booking.user_pk
        LEFT JOIN metro_trips trip
            ON trip.id = payment.metro_trip_pk
        LEFT JOIN registered_users trip_user
            ON trip_user.id = trip.user_pk
        WHERE (
                booking.booking_id = %s
                AND booking_user.email = %s
              )
           OR (
                trip.trip_id = %s
                AND trip_user.email = %s
              )
        ORDER BY payment.paid_at DESC
        LIMIT 1
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (booking_id, user_email, booking_id, user_email))
            row = cur.fetchone()
            return dict(row) if row else None


# ── TRANSACTIONAL OPERATIONS ──────────────────────────────────────────────────

def execute_booking(
    user_id: str,
    schedule_id: str,
    origin_station_id: str,
    destination_station_id: str,
    travel_date: str,
    fare_class: str,
    seat_id: str,
    ticket_type: str = "single",
) -> tuple[bool, dict | str]:
    """
    Create a national rail booking for a logged-in user.

    Args:
        user_id:                e.g. "RU01" — must match the logged-in user
        schedule_id:            e.g. "NR_SCH01"
        origin_station_id:      e.g. "NR01"
        destination_station_id: e.g. "NR05"
        travel_date:            e.g. "2025-06-01"
        fare_class:             "standard" or "first"
        seat_id:                e.g. "B05" (or "any" to auto-assign)
        ticket_type:            "single" (default) or "return"

    Returns:
        (True, booking_dict)   on success
        (False, error_message) on failure
    """
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id
                FROM registered_users
                WHERE user_id = %s AND is_active = TRUE
                """,
                (user_id,),
            )
            user = cur.fetchone()
            if user is None:
                conn.rollback()
                return False, "User not found or inactive"

            cur.execute(
                """
                SELECT
                    schedule.id AS schedule_pk,
                    schedule.first_train_time::text AS departure_time,
                    origin_stop.stop_sequence AS origin_sequence,
                    destination_stop.stop_sequence AS destination_sequence,
                    destination_stop.stop_sequence - origin_stop.stop_sequence AS stops_travelled,
                    origin.id AS origin_pk,
                    destination.id AS destination_pk
                FROM national_rail_schedules schedule
                JOIN national_rail_schedule_stops origin_stop
                    ON origin_stop.national_rail_schedule_pk = schedule.id
                JOIN national_rail_stations origin
                    ON origin.id = origin_stop.national_rail_station_pk
                JOIN national_rail_schedule_stops destination_stop
                    ON destination_stop.national_rail_schedule_pk = schedule.id
                JOIN national_rail_stations destination
                    ON destination.id = destination_stop.national_rail_station_pk
                WHERE schedule.schedule_id = %s
                  AND origin.station_id = %s
                  AND destination.station_id = %s
                  AND origin_stop.stop_sequence < destination_stop.stop_sequence
                """,
                (schedule_id, origin_station_id, destination_station_id),
            )
            route = cur.fetchone()
            if route is None:
                conn.rollback()
                return False, "Schedule does not serve the requested route"

            cur.execute(
                """
                SELECT
                    fare_class,
                    base_fare_usd,
                    per_stop_rate_usd,
                    base_fare_usd + (%s * per_stop_rate_usd) AS total_fare_usd
                FROM national_rail_fares
                WHERE national_rail_schedule_pk = %s
                  AND fare_class = %s
                """,
                (route["stops_travelled"], route["schedule_pk"], fare_class),
            )
            fare = cur.fetchone()
            if fare is None:
                conn.rollback()
                return False, "Fare class is not available for this schedule"

            seat_filter = "seat.seat_id = %s" if seat_id != "any" else "TRUE"
            seat_params: tuple = (
                route["schedule_pk"],
                fare_class,
                travel_date,
            )
            if seat_id != "any":
                seat_params = seat_params + (seat_id,)
            cur.execute(
                f"""
                SELECT
                    seat.id AS seat_pk,
                    seat.seat_id,
                    seat.coach,
                    seat.row_number,
                    seat.seat_column
                FROM national_rail_seats seat
                WHERE seat.national_rail_schedule_pk = %s
                  AND seat.fare_class = %s
                  AND NOT EXISTS (
                      SELECT 1
                      FROM national_rail_bookings booking
                      WHERE booking.national_rail_seat_pk = seat.id
                        AND booking.travel_date = %s
                        AND booking.status <> 'cancelled'
                  )
                  AND {seat_filter}
                ORDER BY seat.coach, seat.row_number, seat.seat_column
                LIMIT 1
                """,
                seat_params,
            )
            seat = cur.fetchone()
            if seat is None:
                conn.rollback()
                return False, "No seats available for this schedule/date/fare class"

            booking_id = _gen_booking_id()
            payment_id = _gen_payment_id()
            cur.execute(
                """
                INSERT INTO national_rail_bookings (
                    booking_id,
                    user_pk,
                    national_rail_schedule_pk,
                    origin_station_pk,
                    destination_station_pk,
                    national_rail_seat_pk,
                    travel_date,
                    departure_time,
                    ticket_type,
                    fare_class,
                    stops_travelled,
                    fare_usd,
                    amount_usd,
                    status,
                    booked_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'confirmed', NOW())
                RETURNING id, booking_id, booked_at
                """,
                (
                    booking_id,
                    user["id"],
                    route["schedule_pk"],
                    route["origin_pk"],
                    route["destination_pk"],
                    seat["seat_pk"],
                    travel_date,
                    route["departure_time"],
                    ticket_type,
                    fare_class,
                    route["stops_travelled"],
                    fare["total_fare_usd"],
                    fare["total_fare_usd"],
                ),
            )
            booking = cur.fetchone()
            cur.execute(
                """
                INSERT INTO payments (
                    payment_id,
                    national_rail_booking_pk,
                    amount_usd,
                    method,
                    status,
                    paid_at
                )
                VALUES (%s, %s, %s, 'ewallet', 'paid', NOW())
                RETURNING payment_id, paid_at
                """,
                (payment_id, booking["id"], fare["total_fare_usd"]),
            )
            payment = cur.fetchone()
            conn.commit()
            return True, {
                "booking_id": booking["booking_id"],
                "payment_id": payment["payment_id"],
                "user_id": user_id,
                "schedule_id": schedule_id,
                "origin_station_id": origin_station_id,
                "destination_station_id": destination_station_id,
                "travel_date": travel_date,
                "departure_time": route["departure_time"],
                "ticket_type": ticket_type,
                "fare_class": fare_class,
                "seat_id": seat["seat_id"],
                "coach": seat["coach"],
                "stops_travelled": route["stops_travelled"],
                "fare_usd": fare["total_fare_usd"],
                "amount_usd": fare["total_fare_usd"],
                "status": "confirmed",
                "booked_at": booking["booked_at"],
                "paid_at": payment["paid_at"],
            }
    except psycopg2.IntegrityError as exc:
        conn.rollback()
        return False, f"Booking could not be created: {exc.diag.message_primary}"
    except Exception as exc:
        conn.rollback()
        return False, str(exc)
    finally:
        conn.close()


def execute_cancellation(booking_id: str, user_id: str) -> tuple[bool, dict | str]:
    """
    Cancel a national rail booking owned by the given user.

    Calculates the refund amount according to the booking's service type:
      - Normal service: RF001 windows (100% / 75% / 50% / 0%)
      - Express service: RF002 windows (100% / 50% / 0%)

    Args:
        booking_id: e.g. "BK001"
        user_id:    must match the booking's user_id

    Returns:
        (True, result_dict)  with refund_amount_usd and policy note
        (False, error_msg)
    """
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    booking.id AS booking_pk,
                    booking.booking_id,
                    booking.fare_usd,
                    booking.amount_usd,
                    booking.status,
                    booking.travel_date,
                    booking.departure_time,
                    schedule.service_type,
                    payment.id AS payment_pk
                FROM national_rail_bookings booking
                JOIN registered_users users
                    ON users.id = booking.user_pk
                JOIN national_rail_schedules schedule
                    ON schedule.id = booking.national_rail_schedule_pk
                LEFT JOIN payments payment
                    ON payment.national_rail_booking_pk = booking.id
                WHERE booking.booking_id = %s
                  AND users.user_id = %s
                FOR UPDATE OF booking
                """,
                (booking_id, user_id),
            )
            booking = cur.fetchone()
            if booking is None:
                conn.rollback()
                return False, "Booking not found for this user"
            if booking["status"] == "cancelled":
                conn.rollback()
                return False, "Booking is already cancelled"

            scheduled_at = datetime.combine(
                booking["travel_date"],
                booking["departure_time"],
                tzinfo=timezone.utc,
            )
            hours_before = (scheduled_at - datetime.now(timezone.utc)).total_seconds() / 3600
            amount = booking["fare_usd"]
            if not isinstance(amount, Decimal):
                amount = Decimal(str(amount))

            if booking["service_type"] == "express":
                if hours_before >= 48:
                    refund_percent, admin_fee, note = Decimal("100"), Decimal("1.00"), "RF002 early cancellation"
                elif hours_before >= 24:
                    refund_percent, admin_fee, note = Decimal("50"), Decimal("1.00"), "RF002 late cancellation"
                else:
                    refund_percent, admin_fee, note = Decimal("0"), Decimal("0.00"), "RF002 no refund"
            else:
                if hours_before >= 48:
                    refund_percent, admin_fee, note = Decimal("100"), Decimal("0.00"), "RF001 early cancellation"
                elif hours_before >= 24:
                    refund_percent, admin_fee, note = Decimal("75"), Decimal("0.50"), "RF001 standard cancellation"
                elif hours_before >= 2:
                    refund_percent, admin_fee, note = Decimal("50"), Decimal("0.50"), "RF001 late cancellation"
                else:
                    refund_percent, admin_fee, note = Decimal("0"), Decimal("0.00"), "RF001 no refund"

            refund_amount = (amount * refund_percent / Decimal("100")) - admin_fee
            if refund_amount < 0:
                refund_amount = Decimal("0.00")
            refund_amount = refund_amount.quantize(Decimal("0.01"))

            cur.execute(
                """
                UPDATE national_rail_bookings
                SET status = 'cancelled'
                WHERE id = %s
                """,
                (booking["booking_pk"],),
            )
            if booking["payment_pk"] and refund_amount > 0:
                cur.execute(
                    """
                    UPDATE payments
                    SET status = 'refunded',
                        refunded_amount_usd = %s,
                        refunded_at = NOW()
                    WHERE id = %s
                    """,
                    (refund_amount, booking["payment_pk"]),
                )
            conn.commit()
            return True, {
                "booking_id": booking["booking_id"],
                "status": "cancelled",
                "refund_amount_usd": refund_amount,
                "policy_note": note,
                "hours_before_departure": round(hours_before, 2),
            }
    except Exception as exc:
        conn.rollback()
        return False, str(exc)
    finally:
        conn.close()


# ── AUTHENTICATION QUERIES ────────────────────────────────────────────────────

def register_user(
    email: str,
    first_name: str,
    surname: str,
    year_of_birth: int,
    password: str,
    secret_question: str,
    secret_answer: str,
) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (True, user_id) on success or (False, error_message) on failure.

    Passwords are stored as Argon2id hashes.
    """
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            user_id = _gen_user_id()
            cur.execute(
                """
                INSERT INTO registered_users (
                    user_id,
                    first_name,
                    surname,
                    email,
                    date_of_birth,
                    registered_at,
                    is_active
                )
                VALUES (%s, %s, %s, %s, %s, NOW(), TRUE)
                RETURNING id
                """,
                (
                    user_id,
                    first_name,
                    surname,
                    email,
                    f"{year_of_birth}-01-01",
                ),
            )
            user_pk = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO user_auth_credentials (
                    user_pk,
                    password_hash,
                    secret_question,
                    secret_answer
                )
                VALUES (%s, %s, %s, %s)
                """,
                (
                    user_pk,
                    PASSWORD_HASHER.hash(password),
                    secret_question,
                    secret_answer,
                ),
            )
            conn.commit()
            return True, user_id
    except psycopg2.IntegrityError as exc:
        conn.rollback()
        return False, exc.diag.message_primary
    except Exception as exc:
        conn.rollback()
        return False, str(exc)
    finally:
        conn.close()


def login_user(email: str, password: str) -> Optional[dict]:
    """
    Verify credentials. Returns a user dict on success or None on failure.
    Dict keys: user_id, email, full_name, first_name, surname, phone, date_of_birth, is_active.
    """
    sql = """
        SELECT
            users.user_id,
            users.email,
            users.first_name || ' ' || users.surname AS full_name,
            users.first_name,
            users.surname,
            users.phone,
            users.date_of_birth,
            users.is_active,
            auth.password_hash
        FROM registered_users users
        JOIN user_auth_credentials auth
            ON auth.user_pk = users.id
        WHERE users.email = %s
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (email,))
            row = cur.fetchone()
            if row is None or not row["is_active"]:
                return None
            try:
                if not PASSWORD_HASHER.verify(row["password_hash"], password):
                    return None
            except (VerifyMismatchError, VerificationError):
                return None
            result = dict(row)
            result.pop("password_hash", None)
            return result


def get_user_secret_question(email: str) -> Optional[str]:
    """Return the secret question for a registered email, or None if not found."""
    sql = """
        SELECT auth.secret_question
        FROM registered_users users
        JOIN user_auth_credentials auth
            ON auth.user_pk = users.id
        WHERE users.email = %s
          AND users.is_active = TRUE
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (email,))
            row = cur.fetchone()
            return row[0] if row else None


def verify_secret_answer(email: str, answer: str) -> bool:
    """Return True if the provided answer matches the stored secret answer (case-insensitive)."""
    sql = """
        SELECT auth.secret_answer
        FROM registered_users users
        JOIN user_auth_credentials auth
            ON auth.user_pk = users.id
        WHERE users.email = %s
          AND users.is_active = TRUE
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (email,))
            row = cur.fetchone()
            if row is None:
                return False
            return row[0].strip().lower() == answer.strip().lower()


def update_password(email: str, new_password: str) -> bool:
    """Update the password for a user. Returns True if the row was updated."""
    sql = """
        UPDATE user_auth_credentials auth
        SET password_hash = %s
        FROM registered_users users
        WHERE users.id = auth.user_pk
          AND users.email = %s
          AND users.is_active = TRUE
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (PASSWORD_HASHER.hash(new_password), email))
            return cur.rowcount > 0


# ── VECTOR / RAG QUERIES — do not modify ─────────────────────────────────────

def query_policy_vector_search(embedding: list[float], top_k: int = VECTOR_TOP_K) -> list[dict]:
    """
    Find the most relevant policy documents for a given query embedding.

    Args:
        embedding: Query vector from llm.embed(user_question)
        top_k:     Number of results to return

    Returns:
        List of dicts with title, category, content, and similarity score
    """
    sql = """
        SELECT
            title,
            category,
            content,
            1 - (embedding <=> %s::vector) AS similarity
        FROM policy_documents
        WHERE 1 - (embedding <=> %s::vector) > %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (vec_str, vec_str, VECTOR_SIMILARITY_THRESHOLD, vec_str, top_k))
            return [dict(row) for row in cur.fetchall()]


def store_policy_document(
    title: str,
    category: str,
    content: str,
    embedding: list[float],
    source_file: str = "",
) -> int:
    """
    Insert a policy document with its embedding into the database.
    Used by skeleton/seed_vectors.py — students don't need to call this directly.

    Returns:
        The new document's id
    """
    sql = """
        INSERT INTO policy_documents (title, category, content, embedding, source_file)
        VALUES (%s, %s, %s, %s::vector, %s)
        RETURNING id
    """
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (title, category, content, vec_str, source_file))
            return cur.fetchone()[0]

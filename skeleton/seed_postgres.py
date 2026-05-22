"""
Seed PostgreSQL with all TransitFlow mock data from train-mock-data/.

Usage:
    python skeleton/seed_postgres.py

Run AFTER docker-compose up -d.
You must first design and create your tables in databases/relational/schema.sql.
Safe to re-run: implement your inserts with ON CONFLICT DO NOTHING.
"""

import json
import os
import sys

import psycopg2
from psycopg2.extras import execute_values
from argon2 import PasswordHasher

# ── resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "train-mock-data")

sys.path.insert(0, PROJECT_DIR)
from skeleton import config as cfg

PASSWORD_HASHER = PasswordHasher()


def load(filename):
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


def connect():
    return psycopg2.connect(
        host=cfg.PG_HOST,
        port=cfg.PG_PORT,
        dbname=cfg.PG_DB,
        user=cfg.PG_USER,
        password=cfg.PG_PASSWORD,
    )


def insert_many(cur, table, columns, rows):
    """Bulk insert with ON CONFLICT DO NOTHING. Returns row count inserted."""
    if not rows:
        return 0
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s "
        f"ON CONFLICT DO NOTHING"
    )
    execute_values(cur, sql, rows)
    return cur.rowcount


def get_id(cur, table, code_column, code_value):
    cur.execute(f"SELECT id FROM {table} WHERE {code_column} = %s", (code_value,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Missing {table}.{code_column}={code_value}")
    return row[0]


def split_full_name(full_name):
    parts = full_name.split(" ", 1)
    first_name = parts[0]
    surname = parts[1] if len(parts) > 1 else ""
    return first_name, surname


# ── seeders ──────────────────────────────────────────────────────────────────

def seed_metro_stations(cur):
    data = load("metro_stations.json")
    rows = [
        (
            item["station_id"],
            item["name"],
            item["is_interchange_metro"],
            item["is_interchange_national_rail"],
        )
        for item in data
    ]
    count = insert_many(
        cur,
        "metro_stations",
        ["station_id", "name", "is_interchange_metro", "is_interchange_national_rail"],
        rows,
    )

    line_rows = []
    interchange_line_rows = []
    for item in data:
        station_pk = get_id(cur, "metro_stations", "station_id", item["station_id"])
        line_rows.extend((station_pk, line) for line in item.get("lines", []))
        interchange_line_rows.extend(
            (station_pk, line) for line in item.get("interchange_metro_lines", [])
        )
    insert_many(cur, "metro_station_lines", ["metro_station_pk", "line"], line_rows)
    insert_many(
        cur,
        "metro_interchange_lines",
        ["metro_station_pk", "line"],
        interchange_line_rows,
    )
    print(f"  metro_stations: {count}")


def seed_national_rail_stations(cur):
    data = load("national_rail_stations.json")
    rows = [
        (
            item["station_id"],
            item["name"],
            item["is_interchange_national_rail"],
            item["is_interchange_metro"],
        )
        for item in data
    ]
    count = insert_many(
        cur,
        "national_rail_stations",
        ["station_id", "name", "is_interchange_national_rail", "is_interchange_metro"],
        rows,
    )

    line_rows = []
    interchange_line_rows = []
    interchange_rows = []
    for item in data:
        station_pk = get_id(cur, "national_rail_stations", "station_id", item["station_id"])
        line_rows.extend((station_pk, line) for line in item.get("lines", []))
        interchange_line_rows.extend(
            (station_pk, line) for line in item.get("interchange_national_rail_lines", [])
        )
        metro_station_id = item.get("interchange_metro_station_id")
        if metro_station_id:
            metro_station_pk = get_id(cur, "metro_stations", "station_id", metro_station_id)
            interchange_rows.append((metro_station_pk, station_pk))

    insert_many(
        cur,
        "national_rail_station_lines",
        ["national_rail_station_pk", "line"],
        line_rows,
    )
    insert_many(
        cur,
        "national_rail_interchange_lines",
        ["national_rail_station_pk", "line"],
        interchange_line_rows,
    )
    insert_many(
        cur,
        "station_interchanges",
        ["metro_station_pk", "national_rail_station_pk"],
        interchange_rows,
    )
    print(f"  national_rail_stations: {count}")


def seed_metro_schedules(cur):
    data = load("metro_schedules.json")
    rows = []
    stop_rows = []
    day_rows = []
    for item in data:
        origin_pk = get_id(cur, "metro_stations", "station_id", item["origin_station_id"])
        destination_pk = get_id(
            cur, "metro_stations", "station_id", item["destination_station_id"]
        )
        rows.append(
            (
                item["schedule_id"],
                item["line"],
                item["direction"],
                origin_pk,
                destination_pk,
                item["first_train_time"],
                item["last_train_time"],
                item["base_fare_usd"],
                item["per_stop_rate_usd"],
                item["frequency_min"],
            )
        )

    count = insert_many(
        cur,
        "metro_schedules",
        [
            "schedule_id",
            "line",
            "direction",
            "origin_station_pk",
            "destination_station_pk",
            "first_train_time",
            "last_train_time",
            "base_fare_usd",
            "per_stop_rate_usd",
            "frequency_min",
        ],
        rows,
    )

    for item in data:
        schedule_pk = get_id(cur, "metro_schedules", "schedule_id", item["schedule_id"])
        for sequence, station_id in enumerate(item["stops_in_order"], start=1):
            station_pk = get_id(cur, "metro_stations", "station_id", station_id)
            stop_rows.append(
                (
                    schedule_pk,
                    sequence,
                    station_pk,
                    item["travel_time_from_origin_min"][station_id],
                )
            )
        day_rows.extend((schedule_pk, day) for day in item["operates_on"])

    insert_many(
        cur,
        "metro_schedule_stops",
        ["metro_schedule_pk", "stop_sequence", "metro_station_pk", "travel_time_from_origin_min"],
        stop_rows,
    )
    insert_many(
        cur,
        "metro_schedule_operating_days",
        ["metro_schedule_pk", "day_of_week"],
        day_rows,
    )
    print(f"  metro_schedules: {count}")


def seed_national_rail_schedules(cur):
    data = load("national_rail_schedules.json")
    rows = []
    stop_rows = []
    day_rows = []
    fare_rows = []
    for item in data:
        origin_pk = get_id(
            cur, "national_rail_stations", "station_id", item["origin_station_id"]
        )
        destination_pk = get_id(
            cur, "national_rail_stations", "station_id", item["destination_station_id"]
        )
        rows.append(
            (
                item["schedule_id"],
                item["line"],
                item["service_type"],
                item["direction"],
                origin_pk,
                destination_pk,
                item["first_train_time"],
                item["last_train_time"],
                item["frequency_min"],
            )
        )

    count = insert_many(
        cur,
        "national_rail_schedules",
        [
            "schedule_id",
            "line",
            "service_type",
            "direction",
            "origin_station_pk",
            "destination_station_pk",
            "first_train_time",
            "last_train_time",
            "frequency_min",
        ],
        rows,
    )

    for item in data:
        schedule_pk = get_id(
            cur, "national_rail_schedules", "schedule_id", item["schedule_id"]
        )
        for sequence, station_id in enumerate(item["stops_in_order"], start=1):
            station_pk = get_id(cur, "national_rail_stations", "station_id", station_id)
            stop_rows.append(
                (
                    schedule_pk,
                    sequence,
                    station_pk,
                    item["travel_time_from_origin_min"][station_id],
                )
            )
        day_rows.extend((schedule_pk, day) for day in item["operates_on"])
        for fare_class, fare in item["fare_classes"].items():
            fare_rows.append(
                (
                    schedule_pk,
                    fare_class,
                    fare["base_fare_usd"],
                    fare["per_stop_rate_usd"],
                )
            )

    insert_many(
        cur,
        "national_rail_schedule_stops",
        [
            "national_rail_schedule_pk",
            "stop_sequence",
            "national_rail_station_pk",
            "travel_time_from_origin_min",
        ],
        stop_rows,
    )
    insert_many(
        cur,
        "national_rail_schedule_operating_days",
        ["national_rail_schedule_pk", "day_of_week"],
        day_rows,
    )
    insert_many(
        cur,
        "national_rail_fares",
        ["national_rail_schedule_pk", "fare_class", "base_fare_usd", "per_stop_rate_usd"],
        fare_rows,
    )
    print(f"  national_rail_schedules: {count}")


def seed_seat_layouts(cur):
    data = load("national_rail_seat_layouts.json")
    rows = []
    for layout in data:
        schedule_pk = get_id(
            cur, "national_rail_schedules", "schedule_id", layout["schedule_id"]
        )
        for coach in layout["coaches"]:
            for seat in coach["seats"]:
                rows.append(
                    (
                        schedule_pk,
                        seat["seat_id"],
                        coach["coach"],
                        coach["fare_class"],
                        seat["row"],
                        seat["column"],
                    )
                )
    count = insert_many(
        cur,
        "national_rail_seats",
        [
            "national_rail_schedule_pk",
            "seat_id",
            "coach",
            "fare_class",
            "row_number",
            "seat_column",
        ],
        rows,
    )
    print(f"  national_rail_seats: {count}")


def seed_users(cur):
    data = load("registered_users.json")
    user_rows = []
    for item in data:
        first_name, surname = split_full_name(item["full_name"])
        user_rows.append(
            (
                item["user_id"],
                first_name,
                surname,
                item["email"],
                item.get("phone"),
                item.get("date_of_birth"),
                item["registered_at"],
                item["is_active"],
            )
        )
    count = insert_many(
        cur,
        "registered_users",
        [
            "user_id",
            "first_name",
            "surname",
            "email",
            "phone",
            "date_of_birth",
            "registered_at",
            "is_active",
        ],
        user_rows,
    )

    auth_rows = []
    for item in data:
        user_pk = get_id(cur, "registered_users", "user_id", item["user_id"])
        auth_rows.append(
            (
                user_pk,
                PASSWORD_HASHER.hash(item["password"]),
                item["secret_question"],
                item["secret_answer"],
            )
        )
    insert_many(
        cur,
        "user_auth_credentials",
        ["user_pk", "password_hash", "secret_question", "secret_answer"],
        auth_rows,
    )
    print(f"  registered_users: {count}")


def seed_national_rail_bookings(cur):
    data = load("bookings.json")
    rows = []
    for item in data:
        user_pk = get_id(cur, "registered_users", "user_id", item["user_id"])
        schedule_pk = get_id(
            cur, "national_rail_schedules", "schedule_id", item["schedule_id"]
        )
        origin_pk = get_id(
            cur, "national_rail_stations", "station_id", item["origin_station_id"]
        )
        destination_pk = get_id(
            cur, "national_rail_stations", "station_id", item["destination_station_id"]
        )
        cur.execute(
            """
            SELECT id
            FROM national_rail_seats
            WHERE national_rail_schedule_pk = %s AND seat_id = %s
            """,
            (schedule_pk, item["seat_id"]),
        )
        seat_row = cur.fetchone()
        if seat_row is None:
            raise ValueError(f"Missing seat {item['seat_id']} for {item['schedule_id']}")
        rows.append(
            (
                item["booking_id"],
                user_pk,
                schedule_pk,
                origin_pk,
                destination_pk,
                seat_row[0],
                item["travel_date"],
                item["departure_time"],
                item["ticket_type"],
                item["fare_class"],
                item["stops_travelled"],
                item["amount_usd"],
                item["status"],
                item["booked_at"],
                item["travelled_at"],
            )
        )
    count = insert_many(
        cur,
        "national_rail_bookings",
        [
            "booking_id",
            "user_pk",
            "national_rail_schedule_pk",
            "origin_station_pk",
            "destination_station_pk",
            "national_rail_seat_pk",
            "travel_date",
            "departure_time",
            "ticket_type",
            "fare_class",
            "stops_travelled",
            "amount_usd",
            "status",
            "booked_at",
            "travelled_at",
        ],
        rows,
    )
    print(f"  national_rail_bookings: {count}")


def seed_metro_travels(cur):
    data = load("metro_travel_history.json")
    rows = []
    for item in data:
        user_pk = get_id(cur, "registered_users", "user_id", item["user_id"])
        schedule_pk = get_id(cur, "metro_schedules", "schedule_id", item["schedule_id"])
        origin_pk = get_id(cur, "metro_stations", "station_id", item["origin_station_id"])
        destination_pk = get_id(
            cur, "metro_stations", "station_id", item["destination_station_id"]
        )
        rows.append(
            (
                item["trip_id"],
                user_pk,
                schedule_pk,
                origin_pk,
                destination_pk,
                item["travel_date"],
                item["ticket_type"],
                item.get("day_pass_ref"),
                item.get("stops_travelled"),
                item["amount_usd"],
                item["status"],
                item["purchased_at"],
                item["travelled_at"],
            )
        )
    count = insert_many(
        cur,
        "metro_trips",
        [
            "trip_id",
            "user_pk",
            "metro_schedule_pk",
            "origin_station_pk",
            "destination_station_pk",
            "travel_date",
            "ticket_type",
            "day_pass_ref",
            "stops_travelled",
            "amount_usd",
            "status",
            "purchased_at",
            "travelled_at",
        ],
        rows,
    )
    print(f"  metro_trips: {count}")


def seed_payments(cur):
    data = load("payments.json")
    rows = []
    for item in data:
        national_rail_booking_pk = None
        metro_trip_pk = None
        if item["booking_id"].startswith("BK"):
            national_rail_booking_pk = get_id(
                cur, "national_rail_bookings", "booking_id", item["booking_id"]
            )
        elif item["booking_id"].startswith("MT"):
            metro_trip_pk = get_id(cur, "metro_trips", "trip_id", item["booking_id"])
        else:
            raise ValueError(f"Unknown payment target {item['booking_id']}")
        rows.append(
            (
                item["payment_id"],
                national_rail_booking_pk,
                metro_trip_pk,
                item["amount_usd"],
                item["method"],
                item["status"],
                item["paid_at"],
            )
        )
    count = insert_many(
        cur,
        "payments",
        [
            "payment_id",
            "national_rail_booking_pk",
            "metro_trip_pk",
            "amount_usd",
            "method",
            "status",
            "paid_at",
        ],
        rows,
    )
    print(f"  payments: {count}")


def seed_feedback(cur):
    data = load("feedback.json")
    rows = []
    for item in data:
        national_rail_booking_pk = None
        metro_trip_pk = None
        if item["booking_id"].startswith("BK"):
            national_rail_booking_pk = get_id(
                cur, "national_rail_bookings", "booking_id", item["booking_id"]
            )
        elif item["booking_id"].startswith("MT"):
            metro_trip_pk = get_id(cur, "metro_trips", "trip_id", item["booking_id"])
        else:
            raise ValueError(f"Unknown feedback target {item['booking_id']}")
        user_pk = get_id(cur, "registered_users", "user_id", item["user_id"])
        rows.append(
            (
                item["feedback_id"],
                national_rail_booking_pk,
                metro_trip_pk,
                user_pk,
                item["rating"],
                item.get("comment"),
                item["submitted_at"],
            )
        )
    count = insert_many(
        cur,
        "feedback",
        [
            "feedback_id",
            "national_rail_booking_pk",
            "metro_trip_pk",
            "user_pk",
            "rating",
            "comment",
            "submitted_at",
        ],
        rows,
    )
    print(f"  feedback: {count}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to PostgreSQL...")
    conn = connect()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("Seeding tables (dependency order):")
        seed_metro_stations(cur)
        seed_national_rail_stations(cur)
        seed_metro_schedules(cur)
        seed_national_rail_schedules(cur)
        seed_seat_layouts(cur)
        seed_users(cur)
        seed_national_rail_bookings(cur)
        seed_metro_travels(cur)
        seed_payments(cur)
        seed_feedback(cur)
        conn.commit()
        print("\nAll done. Database seeded successfully.")
    except Exception as e:
        conn.rollback()
        print(f"\nError: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()

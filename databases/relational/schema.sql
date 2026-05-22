-- ============================================================
--  TransitFlow PostgreSQL Schema
--  Seed data is loaded separately by: python skeleton/seed_postgres.py
--
--  TWO ROLES:
--    1. Relational  → dual-network transit data you design below
--    2. Vector      → policy documents for RAG (provided — do not modify)
-- ============================================================

-- ============================================================
--  STUDENT TASK — Design and create your relational tables here
--
--  Start from the mock data in train-mock-data/:
--    metro_stations.json, national_rail_stations.json
--    metro_schedules.json, national_rail_schedules.json
--    national_rail_seat_layouts.json
--    registered_users.json
--    bookings.json, metro_travel_history.json
--    payments.json, feedback.json
--
--  Think about:
--    - What tables do you need?
--    - What columns and data types?
--    - Which fields are primary keys? Which are foreign keys?
--    - What constraints make sense?
--
--  Apply your schema with:
--    docker-compose down -v && docker-compose up -d
-- ============================================================

-- ============================================================
--  RELATIONAL SCHEMA
--
--  Internal primary keys (`id`) are stable database identifiers.
--  Mock-data IDs such as RU01, MS01, NR_SCH01, BK001, and PM001 are
--  stored as unique external codes so their display format can change.
-- ============================================================

CREATE TABLE IF NOT EXISTS registered_users (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id        VARCHAR(20) NOT NULL UNIQUE,
    first_name     VARCHAR(80) NOT NULL,
    surname        VARCHAR(80) NOT NULL,
    email          VARCHAR(150) NOT NULL UNIQUE,
    phone          VARCHAR(30),
    date_of_birth  DATE,
    registered_at  TIMESTAMPTZ NOT NULL,
    is_active      BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS user_auth_credentials (
    user_pk         BIGINT PRIMARY KEY
        REFERENCES registered_users(id),
    password_hash   TEXT NOT NULL,
    secret_question TEXT NOT NULL,
    secret_answer   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metro_stations (
    id                            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    station_id                    VARCHAR(20) NOT NULL UNIQUE,
    name                          VARCHAR(120) NOT NULL,
    is_interchange_metro          BOOLEAN NOT NULL DEFAULT FALSE,
    is_interchange_national_rail  BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS national_rail_stations (
    id                             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    station_id                     VARCHAR(20) NOT NULL UNIQUE,
    name                           VARCHAR(120) NOT NULL,
    is_interchange_national_rail   BOOLEAN NOT NULL DEFAULT FALSE,
    is_interchange_metro           BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS station_interchanges (
    metro_station_pk         BIGINT NOT NULL
        REFERENCES metro_stations(id),
    national_rail_station_pk BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    PRIMARY KEY (metro_station_pk, national_rail_station_pk)
);

CREATE TABLE IF NOT EXISTS metro_station_lines (
    metro_station_pk BIGINT NOT NULL
        REFERENCES metro_stations(id),
    line             VARCHAR(20) NOT NULL,
    PRIMARY KEY (metro_station_pk, line)
);

CREATE TABLE IF NOT EXISTS metro_interchange_lines (
    metro_station_pk BIGINT NOT NULL
        REFERENCES metro_stations(id),
    line             VARCHAR(20) NOT NULL,
    PRIMARY KEY (metro_station_pk, line)
);

CREATE TABLE IF NOT EXISTS national_rail_station_lines (
    national_rail_station_pk BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    line                     VARCHAR(20) NOT NULL,
    PRIMARY KEY (national_rail_station_pk, line)
);

CREATE TABLE IF NOT EXISTS national_rail_interchange_lines (
    national_rail_station_pk BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    line                     VARCHAR(20) NOT NULL,
    PRIMARY KEY (national_rail_station_pk, line)
);

CREATE TABLE IF NOT EXISTS metro_schedules (
    id                     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    schedule_id            VARCHAR(30) NOT NULL UNIQUE,
    line                   VARCHAR(20) NOT NULL,
    direction              VARCHAR(30) NOT NULL,
    origin_station_pk      BIGINT NOT NULL
        REFERENCES metro_stations(id),
    destination_station_pk BIGINT NOT NULL
        REFERENCES metro_stations(id),
    first_train_time       TIME NOT NULL,
    last_train_time        TIME NOT NULL,
    base_fare_usd          NUMERIC(8, 2) NOT NULL CHECK (base_fare_usd >= 0),
    per_stop_rate_usd      NUMERIC(8, 2) NOT NULL CHECK (per_stop_rate_usd >= 0),
    frequency_min          INTEGER NOT NULL CHECK (frequency_min > 0)
);

CREATE TABLE IF NOT EXISTS metro_schedule_stops (
    metro_schedule_pk           BIGINT NOT NULL
        REFERENCES metro_schedules(id),
    stop_sequence               INTEGER NOT NULL CHECK (stop_sequence > 0),
    metro_station_pk            BIGINT NOT NULL
        REFERENCES metro_stations(id),
    travel_time_from_origin_min INTEGER NOT NULL CHECK (travel_time_from_origin_min >= 0),
    PRIMARY KEY (metro_schedule_pk, stop_sequence),
    UNIQUE (metro_schedule_pk, metro_station_pk)
);

CREATE TABLE IF NOT EXISTS metro_schedule_operating_days (
    metro_schedule_pk BIGINT NOT NULL
        REFERENCES metro_schedules(id),
    day_of_week       VARCHAR(10) NOT NULL CHECK (
        day_of_week IN ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')
    ),
    PRIMARY KEY (metro_schedule_pk, day_of_week)
);

CREATE TABLE IF NOT EXISTS national_rail_schedules (
    id                     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    schedule_id            VARCHAR(30) NOT NULL UNIQUE,
    line                   VARCHAR(20) NOT NULL,
    service_type           VARCHAR(20) NOT NULL CHECK (service_type IN ('normal', 'express')),
    direction              VARCHAR(30) NOT NULL,
    origin_station_pk      BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    destination_station_pk BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    first_train_time       TIME NOT NULL,
    last_train_time        TIME NOT NULL,
    frequency_min          INTEGER NOT NULL CHECK (frequency_min > 0)
);

CREATE TABLE IF NOT EXISTS national_rail_schedule_stops (
    national_rail_schedule_pk   BIGINT NOT NULL
        REFERENCES national_rail_schedules(id),
    stop_sequence               INTEGER NOT NULL CHECK (stop_sequence > 0),
    national_rail_station_pk    BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    travel_time_from_origin_min INTEGER NOT NULL CHECK (travel_time_from_origin_min >= 0),
    PRIMARY KEY (national_rail_schedule_pk, stop_sequence),
    UNIQUE (national_rail_schedule_pk, national_rail_station_pk)
);

CREATE TABLE IF NOT EXISTS national_rail_schedule_operating_days (
    national_rail_schedule_pk BIGINT NOT NULL
        REFERENCES national_rail_schedules(id),
    day_of_week               VARCHAR(10) NOT NULL CHECK (
        day_of_week IN ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')
    ),
    PRIMARY KEY (national_rail_schedule_pk, day_of_week)
);

CREATE TABLE IF NOT EXISTS national_rail_fares (
    national_rail_schedule_pk BIGINT NOT NULL
        REFERENCES national_rail_schedules(id),
    fare_class                VARCHAR(20) NOT NULL CHECK (fare_class IN ('standard', 'first')),
    base_fare_usd             NUMERIC(8, 2) NOT NULL CHECK (base_fare_usd >= 0),
    per_stop_rate_usd         NUMERIC(8, 2) NOT NULL CHECK (per_stop_rate_usd >= 0),
    PRIMARY KEY (national_rail_schedule_pk, fare_class)
);

CREATE TABLE IF NOT EXISTS national_rail_seats (
    id                        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    national_rail_schedule_pk BIGINT NOT NULL
        REFERENCES national_rail_schedules(id),
    seat_id                   VARCHAR(10) NOT NULL,
    coach                     VARCHAR(10) NOT NULL,
    fare_class                VARCHAR(20) NOT NULL CHECK (fare_class IN ('standard', 'first')),
    row_number                INTEGER NOT NULL CHECK (row_number > 0),
    seat_column               VARCHAR(10) NOT NULL,
    UNIQUE (national_rail_schedule_pk, seat_id)
);

CREATE TABLE IF NOT EXISTS national_rail_bookings (
    id                         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    booking_id                 VARCHAR(30) NOT NULL UNIQUE,
    user_pk                    BIGINT NOT NULL
        REFERENCES registered_users(id),
    national_rail_schedule_pk  BIGINT NOT NULL
        REFERENCES national_rail_schedules(id),
    origin_station_pk          BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    destination_station_pk     BIGINT NOT NULL
        REFERENCES national_rail_stations(id),
    national_rail_seat_pk      BIGINT NOT NULL
        REFERENCES national_rail_seats(id),
    travel_date                DATE NOT NULL,
    departure_time             TIME NOT NULL,
    ticket_type                VARCHAR(20) NOT NULL CHECK (ticket_type IN ('single', 'return')),
    fare_class                 VARCHAR(20) NOT NULL CHECK (fare_class IN ('standard', 'first')),
    stops_travelled            INTEGER NOT NULL CHECK (stops_travelled > 0),
    amount_usd                 NUMERIC(8, 2) NOT NULL CHECK (amount_usd >= 0),
    status                     VARCHAR(20) NOT NULL CHECK (status IN ('confirmed', 'completed', 'cancelled')),
    booked_at                  TIMESTAMPTZ NOT NULL,
    travelled_at               TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS metro_trips (
    id                     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trip_id                VARCHAR(30) NOT NULL UNIQUE,
    user_pk                BIGINT NOT NULL
        REFERENCES registered_users(id),
    metro_schedule_pk      BIGINT NOT NULL
        REFERENCES metro_schedules(id),
    origin_station_pk      BIGINT NOT NULL
        REFERENCES metro_stations(id),
    destination_station_pk BIGINT NOT NULL
        REFERENCES metro_stations(id),
    travel_date            DATE NOT NULL,
    ticket_type            VARCHAR(20) NOT NULL CHECK (ticket_type IN ('single', 'day_pass')),
    day_pass_ref           VARCHAR(30),
    stops_travelled        INTEGER CHECK (stops_travelled IS NULL OR stops_travelled > 0),
    amount_usd             NUMERIC(8, 2) NOT NULL CHECK (amount_usd >= 0),
    status                 VARCHAR(20) NOT NULL CHECK (status IN ('completed', 'cancelled')),
    purchased_at           TIMESTAMPTZ,
    travelled_at           TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS payments (
    id                       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    payment_id               VARCHAR(30) NOT NULL UNIQUE,
    national_rail_booking_pk BIGINT
        REFERENCES national_rail_bookings(id),
    metro_trip_pk            BIGINT
        REFERENCES metro_trips(id),
    amount_usd               NUMERIC(8, 2) NOT NULL CHECK (amount_usd >= 0),
    method                   VARCHAR(30) NOT NULL CHECK (method IN ('credit_card', 'debit_card', 'ewallet')),
    status                   VARCHAR(20) NOT NULL CHECK (status IN ('paid', 'refunded', 'failed')),
    paid_at                  TIMESTAMPTZ NOT NULL,
    CHECK (
        (national_rail_booking_pk IS NOT NULL AND metro_trip_pk IS NULL)
        OR
        (national_rail_booking_pk IS NULL AND metro_trip_pk IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS feedback (
    id                       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    feedback_id              VARCHAR(30) NOT NULL UNIQUE,
    national_rail_booking_pk BIGINT
        REFERENCES national_rail_bookings(id),
    metro_trip_pk            BIGINT
        REFERENCES metro_trips(id),
    user_pk                  BIGINT NOT NULL
        REFERENCES registered_users(id),
    rating                   INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment                  TEXT,
    submitted_at             TIMESTAMPTZ NOT NULL,
    CHECK (
        (national_rail_booking_pk IS NOT NULL AND metro_trip_pk IS NULL)
        OR
        (national_rail_booking_pk IS NULL AND metro_trip_pk IS NOT NULL)
    )
);

-- ============================================================
--  RELATIONAL INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_station_interchanges_metro
ON station_interchanges (metro_station_pk);

CREATE INDEX IF NOT EXISTS idx_station_interchanges_national_rail
ON station_interchanges (national_rail_station_pk);

CREATE INDEX IF NOT EXISTS idx_metro_schedule_stops_station
ON metro_schedule_stops (metro_station_pk, metro_schedule_pk, stop_sequence);

CREATE INDEX IF NOT EXISTS idx_national_rail_schedule_stops_station
ON national_rail_schedule_stops (national_rail_station_pk, national_rail_schedule_pk, stop_sequence);

CREATE INDEX IF NOT EXISTS idx_national_rail_bookings_user_date
ON national_rail_bookings (user_pk, travel_date DESC);

CREATE INDEX IF NOT EXISTS idx_metro_trips_user_date
ON metro_trips (user_pk, travel_date DESC);

CREATE INDEX IF NOT EXISTS idx_national_rail_bookings_schedule_date_status
ON national_rail_bookings (national_rail_schedule_pk, travel_date, status);

CREATE INDEX IF NOT EXISTS idx_national_rail_seats_schedule_class
ON national_rail_seats (national_rail_schedule_pk, fare_class);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_national_rail_seat_booking
ON national_rail_bookings (national_rail_schedule_pk, travel_date, national_rail_seat_pk)
WHERE status <> 'cancelled';

CREATE INDEX IF NOT EXISTS idx_payments_national_rail_booking
ON payments (national_rail_booking_pk)
WHERE national_rail_booking_pk IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_payments_metro_trip
ON payments (metro_trip_pk)
WHERE metro_trip_pk IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_feedback_national_rail_booking
ON feedback (national_rail_booking_pk)
WHERE national_rail_booking_pk IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_feedback_metro_trip
ON feedback (metro_trip_pk)
WHERE metro_trip_pk IS NOT NULL;




-- ============================================================
--  VECTOR SCHEMA  (RAG / Help Desk) — do not modify
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_documents (
    id          SERIAL       PRIMARY KEY,
    title       VARCHAR(200) NOT NULL,
    category    VARCHAR(50)  NOT NULL,  -- 'refund', 'booking', 'conduct'
    content     TEXT         NOT NULL,
    -- 768-dim  → Ollama nomic-embed-text (default)
    -- 3072-dim → Gemini gemini-embedding-001
    -- If you switch LLM_PROVIDER to gemini, change to vector(3072) and reset the database.
    embedding   vector(768),
    source_file VARCHAR(200),
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);

-- Index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_policy_documents_embedding
ON policy_documents USING hnsw (embedding vector_cosine_ops);

# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2026-05-28 14:29:40
# @Last Modified by:   Your name
# @Last Modified time: 2026-05-29 15:09:31
"""
TransitFlow - Intelligent Agent

This is the brain of the system.

HOW IT WORKS:
1. User asks a natural language question.
2. The LLM decides which database tools to query.
3. Each database query runs and returns structured data.
4. The LLM reads the data and writes a helpful answer.
5. The answer is returned to the Gradio UI.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from typing import Optional

from databases.graph.queries import (
    query_alternative_routes,
    query_cheapest_route,
    query_delay_ripple,
    query_interchange_path,
    query_shortest_route,
)
from databases.relational.queries import (
    execute_booking,
    execute_cancellation,
    query_available_seats,
    query_metro_fare,
    query_metro_schedules,
    query_national_rail_availability,
    query_national_rail_fare,
    query_national_rail_schedule_fares,
    query_payment_info,
    query_policy_vector_search,
    query_user_bookings,
    query_user_profile,
)
from skeleton.llm_provider import llm


# Cache embedded confirmation examples so repeated booking confirmations are faster.
_CONFIRMATION_VECTOR_CACHE: dict[str, list[tuple[str, list[float]]]] = {}

# Example phrases used by the vector fallback to detect confirm/reject intent.
_CONFIRMATION_EXAMPLES = {
    "confirm": [
        "yes, confirm this booking",
        "ok, book it",
        "go ahead and make the booking",
        "submit the booking",
        "I agree to book this ticket",
        "確認訂票",
        "幫我訂這張票",
        "可以，送出訂票",
        "好的，就訂這個",
        "沒問題，請幫我完成訂票",
    ],
    "reject": [
        "no, do not book it",
        "cancel this booking",
        "do not submit the booking",
        "wait, I do not want to book yet",
        "stop the booking",
        "不要訂票",
        "取消訂票",
        "先不要送出",
        "等一下，不要訂",
        "我還不想確認",
    ],
}


# Maps station names to database IDs so natural-language station names can be
# converted into reliable query parameters before tool routing.
_STATION_INDEX: dict[str, str] = {
    # Metro - English
    "central square": "MS01",
    "riverside": "MS02",
    "northgate": "MS03",
    "elm park": "MS04",
    "westfield": "MS05",
    "harbour view": "MS06",
    "old town": "MS07",
    "university": "MS08",
    "queensbridge": "MS09",
    "parkside": "MS10",
    "greenhill": "MS11",
    "lakeshore": "MS12",
    "clifton": "MS13",
    "eastwick": "MS14",
    "ferndale": "MS15",
    "hilltop": "MS16",
    "broadmoor": "MS17",
    "sunnyvale": "MS18",
    "redwood": "MS19",
    "thornton": "MS20",
    # Metro - Chinese
    "中央廣場": "MS01",
    "河濱站": "MS02",
    "北門站": "MS03",
    "榆樹公園站": "MS04",
    "西田站": "MS05",
    "海港景站": "MS06",
    "舊城站": "MS07",
    "大學站": "MS08",
    "皇后橋站": "MS09",
    "公園側站": "MS10",
    "綠丘站": "MS11",
    "湖岸站": "MS12",
    "克利夫頓站": "MS13",
    "東威克站": "MS14",
    "芬戴爾站": "MS15",
    "山頂站": "MS16",
    "寬地站": "MS17",
    "陽光谷站": "MS18",
    "紅木站": "MS19",
    "桑頓站": "MS20",
    # National Rail - English
    "central station": "NR01",
    "maplewood": "NR02",
    "old town junction": "NR03",
    "ashford": "NR04",
    "stonehaven": "NR05",
    "bridgeport": "NR06",
    "ferndale halt": "NR07",
    "coalport": "NR08",
    "dunmore": "NR09",
    "langford end": "NR10",
    # National Rail - Chinese
    "中央站": "NR01",
    "楓木站": "NR02",
    "舊城交匯站": "NR03",
    "阿什福德站": "NR04",
    "石港站": "NR05",
    "橋港站": "NR06",
    "芬戴爾停靠站": "NR07",
    "煤港站": "NR08",
    "丹摩站": "NR09",
    "蘭福德終點站": "NR10",
}


def _inject_station_ids(text: str) -> str:
    """
    Replace station names in text with 'name (ID)' so the LLM reads the ID
    right next to the name and uses it as the parameter value.
    """
    result = text
    seen_ids: set[str] = set()
    for name in sorted(_STATION_INDEX, key=len, reverse=True):
        sid = _STATION_INDEX[name]
        if sid in seen_ids:
            continue
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(f"{name} ({sid})", result)
            seen_ids.add(sid)
    return result


# Main instruction prompt for the final answer-writing LLM call.
SYSTEM_PROMPT = """You are TransitFlow, a friendly and patient transit assistant for a dual-network system.

Networks: City Metro MS01-MS20 (lines M1-M4) | National Rail NR01-NR10 (lines NR1-NR2)
Interchanges: Central=MS01/NR01 | Old Town=MS07/NR03 | Ferndale=MS15/NR07
Today: {today}

PERSONALITY:
- Be warm, helpful, and patient like a real customer service agent.
- Never show raw error codes or technical messages to the user.
- When data is not found, apologise and suggest next steps.
- Always end with an offer to help further.

RESPONSE FORMAT:
- Use emojis to make responses easier to read.
- For schedules, use clear sections with labels.
- For routes, list every station in order and highlight interchange points.
- Keep responses concise but complete.

BOOKING CONFIRMATION RULE:
- Before calling make_booking, ALWAYS summarise the booking details and ask the user to confirm.
- Only call make_booking after the user explicitly says "confirm", "yes", "確認", "好", or "ok".

LOGIN RULE: Routes, fares, schedules, and policies work WITHOUT login.
Only make_booking and cancel_booking need login.

When DATA FROM TRANSITFLOW DATABASE is provided, use it as the only source of truth.
For route results: list every station name in order, note any line changes, and give the total travel time.
Always reply in the same language as the user.
""".format(today=date.today().isoformat())


# Tool metadata used by the native tool router / JSON router.
TOOLS = [
    {
        "name": "check_national_rail_availability",
        "description": (
            "Check available national rail trains and services between two stations. "
            "Use for schedules, timetables, or availability."
        ),
        "parameters": {
            "origin_id": {"type": "string", "description": "National rail station ID e.g. NR01"},
            "destination_id": {"type": "string", "description": "National rail station ID e.g. NR05"},
            "travel_date": {"type": "string", "description": "YYYY-MM-DD, optional"},
        },
        "required": ["origin_id", "destination_id"],
    },
    {
        "name": "get_national_rail_fare",
        "description": "Calculate the fare for a national rail journey on a specific schedule.",
        "parameters": {
            "schedule_id": {"type": "string", "description": "e.g. NR_SCH01"},
            "fare_class": {"type": "string", "description": "standard or first"},
            "stops_travelled": {"type": "integer", "description": "Number of stops"},
        },
        "required": ["schedule_id", "fare_class", "stops_travelled"],
    },
    {
        "name": "get_national_rail_schedule_fares",
        "description": (
            "Get all ticket prices for a national rail schedule id when the user asks "
            "for the price of a service such as NR_SCH04."
        ),
        "parameters": {
            "schedule_id": {"type": "string", "description": "e.g. NR_SCH04"},
        },
        "required": ["schedule_id"],
    },
    {
        "name": "check_metro_availability",
        "description": "Check available metro services between two metro stations.",
        "parameters": {
            "origin_id": {"type": "string", "description": "Metro station ID e.g. MS01"},
            "destination_id": {"type": "string", "description": "Metro station ID e.g. MS09"},
        },
        "required": ["origin_id", "destination_id"],
    },
    {
        "name": "calculate_metro_fare",
        "description": "Calculate the metro single-ticket fare for a journey.",
        "parameters": {
            "schedule_id": {"type": "string", "description": "e.g. MS_SCH01"},
            "stops_travelled": {"type": "integer", "description": "Number of stops"},
        },
        "required": ["schedule_id", "stops_travelled"],
    },
    {
        "name": "get_metro_fare",
        "description": (
            "Get the metro ticket price between two stations. "
            "Use for fare/price/cost questions, not route questions."
        ),
        "parameters": {
            "origin_id": {"type": "string", "description": "Metro station ID e.g. MS01"},
            "destination_id": {"type": "string", "description": "Metro station ID e.g. MS09"},
        },
        "required": ["origin_id", "destination_id"],
    },
    {
        "name": "get_user_bookings",
        "description": "Retrieve the logged-in user's full booking history.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "get_user_profile",
        "description": "Retrieve the logged-in user's profile information.",
        "parameters": {},
        "required": [],
    },
    {
        "name": "get_payment_info",
        "description": "Retrieve payment details for a specific booking or metro trip.",
        "parameters": {
            "booking_id": {"type": "string", "description": "Booking or trip ID e.g. BK-A1B2C3"},
        },
        "required": ["booking_id"],
    },
    {
        "name": "get_available_seats",
        "description": "Show available seats on a national rail service.",
        "parameters": {
            "schedule_id": {"type": "string", "description": "e.g. NR_SCH01"},
            "travel_date": {"type": "string", "description": "YYYY-MM-DD"},
            "fare_class": {"type": "string", "description": "standard or first"},
        },
        "required": ["schedule_id", "travel_date", "fare_class"],
    },
    {
        "name": "make_booking",
        "description": "Create a national rail booking for the logged-in user after confirmation.",
        "parameters": {
            "schedule_id": {"type": "string", "description": "e.g. NR_SCH01"},
            "origin_station_id": {"type": "string", "description": "e.g. NR01"},
            "destination_station_id": {"type": "string", "description": "e.g. NR05"},
            "travel_date": {"type": "string", "description": "YYYY-MM-DD"},
            "fare_class": {"type": "string", "description": "standard or first"},
            "seat_id": {"type": "string", "description": "Seat ID or any"},
            "ticket_type": {"type": "string", "description": "single or return"},
        },
        "required": [
            "schedule_id",
            "origin_station_id",
            "destination_station_id",
            "travel_date",
            "fare_class",
            "seat_id",
        ],
    },
    {
        "name": "cancel_booking",
        "description": "Cancel a national rail booking for the logged-in user.",
        "parameters": {
            "booking_id": {"type": "string", "description": "Booking reference e.g. BK-A1B2C3"},
        },
        "required": ["booking_id"],
    },
    {
        "name": "search_policy",
        "description": "Search company policy documents.",
        "parameters": {
            "query": {"type": "string", "description": "Natural language policy question"},
        },
        "required": ["query"],
    },
    {
        "name": "find_route",
        "description": "Find the best route or path between two stations.",
        "parameters": {
            "origin_id": {"type": "string", "description": "Station ID e.g. MS01 or NR01"},
            "destination_id": {"type": "string", "description": "Station ID e.g. MS09 or NR05"},
            "network": {"type": "string", "description": "metro, rail, or auto"},
            "optimise_by": {"type": "string", "description": "time or cost"},
        },
        "required": ["origin_id", "destination_id"],
    },
    {
        "name": "find_alternative_routes",
        "description": "Find routes that avoid a specific delayed or closed station.",
        "parameters": {
            "origin_id": {"type": "string", "description": "e.g. NR01"},
            "destination_id": {"type": "string", "description": "e.g. NR05"},
            "avoid_station_id": {"type": "string", "description": "e.g. NR03"},
            "network": {"type": "string", "description": "metro, rail, or auto"},
        },
        "required": ["origin_id", "destination_id", "avoid_station_id"],
    },
    {
        "name": "get_delay_ripple",
        "description": "Show affected stations and lines from a disruption.",
        "parameters": {
            "station_id": {"type": "string", "description": "Station ID e.g. NR03 or MS07"},
            "hops": {"type": "integer", "description": "Number of hops"},
        },
        "required": ["station_id"],
    },
]

# Compact tool list injected into the JSON-only router prompt.
TOOLS_SCHEMA = """\
find_route(origin_id, destination_id, optimise_by?)
check_national_rail_availability(origin_id, destination_id, travel_date?)
get_national_rail_fare(schedule_id, fare_class, stops_travelled)
get_national_rail_schedule_fares(schedule_id)
check_metro_availability(origin_id, destination_id)
calculate_metro_fare(schedule_id, stops_travelled)
get_available_seats(schedule_id, travel_date, fare_class)
make_booking(schedule_id, origin_station_id, destination_station_id, travel_date, fare_class, seat_id, ticket_type?)
cancel_booking(booking_id)
get_user_bookings()
get_user_profile()
get_payment_info(booking_id)
search_policy(query)
find_alternative_routes(origin_id, destination_id, avoid_station_id, network?)
get_delay_ripple(station_id, hops?)"""


def _execute_tool(
    tool_name: str,
    params: dict,
    current_user_email: Optional[str] = None,
) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        if tool_name == "check_national_rail_availability":
            # Read PostgreSQL rail schedules and seat availability for a route/date.
            result = query_national_rail_availability(**params)

        elif tool_name == "get_national_rail_fare":
            # If stops are missing/invalid, fall back to showing all fares for the schedule.
            stops_travelled = params.get("stops_travelled")
            try:
                params["stops_travelled"] = int(stops_travelled)
            except (TypeError, ValueError):
                result = query_national_rail_schedule_fares(params["schedule_id"])
                if params.get("fare_class"):
                    result = [
                        row for row in result
                        if str(row.get("fare_class")) == str(params["fare_class"])
                    ]
            else:
                result = query_national_rail_fare(**params)

        elif tool_name == "get_national_rail_schedule_fares":
            # Return every fare class / ticket price configured for a rail schedule.
            result = query_national_rail_schedule_fares(params["schedule_id"])
            if not result:
                return json.dumps({"error": f"找不到班次 {params['schedule_id']} 的票價資料。"})

        elif tool_name == "check_metro_availability":
            # Read PostgreSQL metro schedule records for the requested stations.
            result = query_metro_schedules(
                origin_id=params["origin_id"],
                destination_id=params["destination_id"],
            )

        elif tool_name == "calculate_metro_fare":
            # Calculate fare from a known metro schedule and stop count.
            result = query_metro_fare(**params)

        elif tool_name == "get_metro_fare":
            # Prefer a direct PostgreSQL schedule, then estimate from Neo4j route legs.
            schedules = query_metro_schedules(
                origin_id=params["origin_id"],
                destination_id=params["destination_id"],
            )
            if not schedules:
                route = query_shortest_route(
                    origin_id=params["origin_id"],
                    destination_id=params["destination_id"],
                    network="metro",
                )
                if not route.get("found"):
                    result = {
                        "error": "很抱歉，找不到這兩站之間的捷運服務。請確認站點代碼是否正確。"
                    }
                else:
                    result = {
                        "origin": route["path"][0]["name"],
                        "destination": route["path"][-1]["name"],
                        "fare_source": "graph_route_estimate",
                        "note": (
                            "No direct metro schedule was found in PostgreSQL; "
                            "this fare is estimated from the Neo4j route legs."
                        ),
                        "total_fare_usd": route["total_fare_usd"],
                        "total_time_min": route["total_time_min"],
                        "stations": route["stations"],
                        "legs": route["legs"],
                    }
            else:
                sched = schedules[0]
                stops_travelled = sched.get("stops_travelled", 1)
                fare = query_metro_fare(sched["schedule_id"], stops_travelled)
                result = {
                    "origin": sched.get("origin_name", params["origin_id"]),
                    "destination": sched.get("destination_name", params["destination_id"]),
                    "fare_source": "postgres_direct_schedule",
                    "line": sched.get("line"),
                    "schedule_id": sched["schedule_id"],
                    "stops_travelled": stops_travelled,
                    **(fare or {"error": "很抱歉，票價查詢失敗，請稍後再試。"}),
                }

        elif tool_name == "get_user_bookings":
            # Personal booking history is scoped to the logged-in email.
            if not current_user_email:
                return json.dumps({"error": "您尚未登入。請點右上角的登入按鈕後再試。"})
            result = query_user_bookings(current_user_email)
            print("current_user_email =", current_user_email)
            print("result =", result) 

        elif tool_name == "get_user_profile":
            # Profile lookup verifies the current session user before returning account data.
            if not current_user_email:
                return json.dumps({"error": "您尚未登入。請點右上角的登入按鈕後再試。"})
            result = query_user_profile(current_user_email)
            if result is None:
                return json.dumps({"error": "找不到使用者資料，請重新登入。"})

        elif tool_name == "get_payment_info":
            if not current_user_email:
                return json.dumps({"error": "您尚未登入。請點右上角的登入按鈕後再試。"})
            # Pass the logged-in email so payment lookup stays scoped to the current user.
            result = query_payment_info(params["booking_id"], current_user_email)
            if result is None:
                return json.dumps({"error": f"找不到訂單 {params['booking_id']} 的付款紀錄。"})

        elif tool_name == "get_available_seats":
            # Seat lookup is used before booking so the user can pick an available seat.
            result = query_available_seats(**params)

        elif tool_name == "make_booking":
            # Booking writes require login; email is converted to user_id before inserting.
            if not current_user_email:
                return json.dumps({"error": "您需要先登入才能訂票。"})
            profile = query_user_profile(current_user_email)
            if not profile:
                return json.dumps({"error": "找不到使用者資料，請重新登入。"})
            ok, data = execute_booking(
                user_id=profile["user_id"],
                schedule_id=params["schedule_id"],
                origin_station_id=params["origin_station_id"],
                destination_station_id=params["destination_station_id"],
                travel_date=params["travel_date"],
                fare_class=params["fare_class"],
                seat_id=params["seat_id"],
                ticket_type=params.get("ticket_type", "single"),
            )
            result = data if ok else {"error": f"訂票失敗：{data}"}

        elif tool_name == "cancel_booking":
            # Cancellation also checks the logged-in user's user_id for ownership safety.
            if not current_user_email:
                return json.dumps({"error": "您需要先登入才能取消訂票。"})
            profile = query_user_profile(current_user_email)
            if not profile:
                return json.dumps({"error": "找不到使用者資料，請重新登入。"})
            ok, data = execute_cancellation(
                booking_id=params["booking_id"],
                user_id=profile["user_id"],
            )
            result = data if ok else {"error": f"取消失敗：{data}"}

        elif tool_name == "search_policy":
            # Policy search embeds the question and performs vector similarity search.
            embedding = llm.embed(params["query"])
            docs = query_policy_vector_search(embedding)
            if not docs:
                return json.dumps({"error": "很抱歉，找不到相關政策資訊。請嘗試用不同的關鍵字搜尋。"})
            result = [
                {
                    "title": d["title"],
                    "category": d["category"],
                    "content": d["content"][:800],
                    "similarity": round(d["similarity"], 3),
                }
                for d in docs
            ]

        elif tool_name == "find_route":
            # Route search uses Neo4j and picks interchange, cheapest, or shortest logic.
            origin_id = params["origin_id"]
            destination_id = params["destination_id"]
            network = params.get("network", "auto")
            optimise_by = params.get("optimise_by", "time")

            is_cross = (
                origin_id.upper().startswith("MS")
                and destination_id.upper().startswith("NR")
            ) or (
                origin_id.upper().startswith("NR")
                and destination_id.upper().startswith("MS")
            )
            if is_cross:
                result = query_interchange_path(origin_id, destination_id)
            elif optimise_by == "cost":
                result = query_cheapest_route(
                    origin_id=origin_id,
                    destination_id=destination_id,
                    network=network,
                )
            else:
                result = query_shortest_route(
                    origin_id=origin_id,
                    destination_id=destination_id,
                    network=network,
                )

        elif tool_name == "find_alternative_routes":
            # Alternative routes ask Neo4j for paths that avoid a disrupted station.
            routes = query_alternative_routes(
                origin_id=params["origin_id"],
                destination_id=params["destination_id"],
                avoid_station_id=params["avoid_station_id"],
                network=params.get("network", "auto"),
            )
            result = [{"route_number": i + 1, "legs": r} for i, r in enumerate(routes)]

        elif tool_name == "get_delay_ripple":
            # Delay ripple expands outward from one station to show nearby affected nodes.
            result = query_delay_ripple(
                delayed_station_id=params["station_id"],
                hops=params.get("hops", 2),
            )

        else:
            result = {"error": f"很抱歉，發生未知錯誤（{tool_name}）。請稍後再試。"}

        return json.dumps(result, default=str)

    except Exception as e:
        return json.dumps({"error": f"很抱歉，系統發生錯誤：{str(e)}。請稍後再試。"})


def _flatten_to_text(obj, depth: int = 0) -> str:
    """Recursively convert any JSON value to indented key-value text."""
    pad = "  " * depth
    if isinstance(obj, dict):
        if not obj:
            return f"{pad}(empty)"
        lines = []
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                inner = _flatten_to_text(v, depth + 1)
                if inner.strip():
                    lines.append(f"{pad}{k}:\n{inner}")
            else:
                lines.append(f"{pad}{k}: {v}")
        return "\n".join(lines) or f"{pad}(empty)"
    if isinstance(obj, list):
        if not obj:
            return f"{pad}(no records)"
        parts = []
        for i, item in enumerate(obj, 1):
            if isinstance(item, (dict, list)):
                parts.append(f"{pad}[{i}]")
                parts.append(_flatten_to_text(item, depth + 1))
            else:
                parts.append(f"{pad}- {item}")
        return "\n".join(parts)
    return f"{pad}{obj}"


def _normalise_result(tool_name: str, result_json: str) -> str:
    """Turn tool JSON into readable debug text for the UI debug panel."""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return result_json
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    return _flatten_to_text(data)


def _summarise_result(tool_name: str, result_json: str) -> str:
    """Keep the full JSON payload available for the final LLM answer step."""
    return result_json


def _parse_tool_calls(llm_response: str) -> list[dict] | None:
    """Extract the router's JSON tool_calls object from plain or fenced text."""
    text = llm_response.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if "tool_calls" in data:
            return data["tool_calls"]
    return None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _confirmation_example_vectors(label: str) -> list[tuple[str, list[float]]]:
    """Embed and cache confirmation examples for semantic fallback matching."""
    if label not in _CONFIRMATION_VECTOR_CACHE:
        _CONFIRMATION_VECTOR_CACHE[label] = [
            (example, llm.embed(example))
            for example in _CONFIRMATION_EXAMPLES[label]
        ]
    return _CONFIRMATION_VECTOR_CACHE[label]


def _vector_confirmation_state(message: str) -> str:
    """
    Classify confirmation intent semantically.

    Returns confirm, reject, or unclear. This is only a fallback after explicit
    keyword checks, so it uses conservative thresholds.
    """
    try:
        message_vector = llm.embed(message)
        scores: dict[str, float] = {}
        for label in ("confirm", "reject"):
            example_scores = [
                _cosine_similarity(message_vector, example_vector)
                for _, example_vector in _confirmation_example_vectors(label)
            ]
            scores[label] = max(example_scores, default=0.0)
    except Exception:
        return "unclear"

    confirm_score = scores.get("confirm", 0.0)
    reject_score = scores.get("reject", 0.0)
    margin = abs(confirm_score - reject_score)
    if confirm_score >= 0.78 and confirm_score > reject_score and margin >= 0.04:
        return "confirm"
    if reject_score >= 0.78 and reject_score > confirm_score and margin >= 0.04:
        return "reject"
    return "unclear"


def _user_confirmed(history: list[dict]) -> bool:
    """Check if the most recent user message contains an explicit confirmation."""
    if not history:
        return False
    last_user = next(
        (m["content"].lower() for m in reversed(history) if m["role"] == "user"),
        "",
    )
    reject_words = {
        "no",
        "cancel",
        "不要",
        "先不要",
        "不確認",
        "取消",
        "等等",
        "等一下",
    }
    if any(word in last_user for word in reject_words):
        return False

    if re.search(r"\b(confirm|yes|ok|sure|go ahead)\b", last_user):
        return True

    confirm_words = {
        "確認",
        "好",
        "好的",
        "可以",
        "沒問題",
        "沒錯",
        "對",
        "是的",
        "幫我訂",
        "訂吧",
        "訂了",
        "送出",
    }
    if any(word in last_user for word in confirm_words):
        return True

    return _vector_confirmation_state(last_user) == "confirm"


def run_agent(
    user_message: str,
    history: list[dict],
    debug: bool = False,
    current_user_email: Optional[str] = None,
) -> tuple:
    """Main agent loop."""
    debug_info = []

    # Add login context to the system prompt so personal tools know who is asking.
    if current_user_email:
        profile = query_user_profile(current_user_email)
        if profile:
            user_display = f"{profile['full_name']} (email: {current_user_email}, user_id: {profile['user_id']})"
        else:
            user_display = current_user_email
        contextual_prompt = SYSTEM_PROMPT + (
            f"\n\n目前登入使用者：{user_display}。"
            "請直接回答此使用者的個人訂票問題，不需要再詢問 email 或 ID。"
            "查詢訂票紀錄請使用 get_user_bookings()。"
            "訂票和取消請使用 make_booking / cancel_booking。"
        )
    else:
        contextual_prompt = SYSTEM_PROMPT + (
            "\n\n目前沒有使用者登入。"
            "如果使用者詢問個人訂票、歷史紀錄，或想要訂票、取消訂票，"
            "請友善地告知他們需要先登入。"
        )

    # Keep only recent chat history for routing, then annotate station names with IDs.
    recent_history = history[-4:] if len(history) > 4 else history
    augmented_message = _inject_station_ids(user_message)

    # Ask the LLM router to choose database tools and parameters in strict JSON.
    tool_selection_prompt = f"""Output only this JSON (no other text):
{{"tool_calls": [{{"name": "TOOL", "params": {{"KEY": "VALUE"}}}}]}}
Or if no tool needed: {{"tool_calls": []}}

STATIONS: Metro=MS01-MS20, Rail=NR01-NR10
USER: {current_user_email or "not logged in"}
get_user_bookings: call when logged-in user asks about their bookings, tickets, or travel history.
get_user_profile: call when logged-in user asks about their account or profile.
get_payment_info: call with booking_id when user asks about payment for a specific booking.
make_booking/cancel_booking: only if user is logged in AND has explicitly confirmed.
Route/path/journey/怎麼去/如何前往/路線 questions: use find_route.
Policy/rules/退款/補償/行李/寵物 questions: use search_policy.
Never use "" as a param value. Omit optional params if unknown.

TOOLS:
{TOOLS_SCHEMA}

HISTORY:
{json.dumps(recent_history, indent=None)}

USER: "{augmented_message}"

JSON:"""

    if llm.get_chat_provider() == "ollama":
        # Ollama supports the native tool-call path used by this project.
        try:
            tool_calls = llm.ollama_tool_call(
                recent_history,
                TOOLS,
                augmented_message,
                system_prompt=(
                    "You are a tool router. Call the right tool based on the user message. "
                    f"Logged-in user: {current_user_email or 'none'}. "
                    "My bookings/tickets/travel history/我的訂票 -> get_user_bookings. "
                    "My account/profile/我的帳號 -> get_user_profile. "
                    "Payment info for booking -> get_payment_info. "
                    "Fare/price/cost for a rail schedule id like NR_SCH04 -> get_national_rail_schedule_fares. "
                    "Book a ticket -> check availability first, then make_booking only after confirmation. "
                    "Cancel a booking -> cancel_booking. "
                    "Policy/rules/refund/luggage/bicycle/退款/補償/行李/寵物 -> search_policy. "
                    "Route/directions/fastest/how-to-get/怎麼去/路線 -> find_route. "
                    "Metro fare/price/cost/票價/多少錢 -> get_metro_fare. "
                    "Schedule/timetable/trains/services/班次/時刻表 -> availability tools. "
                    "Only call a tool when needed."
                ),
            )
        except ConnectionError as exc:
            tool_calls = []
            if debug:
                debug_info.append(f"**Tool selection unavailable:** {exc}")
        if debug:
            debug_info.append(f"**Tool selection (native):** {tool_calls}")
    else:
        # Non-Ollama providers use a JSON-only prompt and a lightweight parser.
        selection_response = llm.chat(
            messages=[{"role": "user", "content": tool_selection_prompt}],
            system_prompt="JSON only. You are a router. Output valid JSON. No empty string param values.",
        )
        tool_calls = _parse_tool_calls(selection_response) or []
        if debug:
            debug_info.append(f"**Tool selection:** {selection_response}")

    # Pre-compute common IDs and keywords for deterministic fallback routing.
    lower = augmented_message.lower()
    raw_station_ids = re.findall(r"\b(MS\d{2}|NR\d{2})\b", augmented_message, re.IGNORECASE)
    station_ids = []
    for sid in raw_station_ids:
        sid = sid.upper()
        if sid not in station_ids:
            station_ids.append(sid)
    schedule_ids = re.findall(r"\b(NR_SCH\d{2}|MS_SCH\d{2})\b", augmented_message, re.IGNORECASE)
    two_stations = len(station_ids) >= 2

    def _tool_selected(name: str, *required_params) -> bool:
        """Check whether the router already selected a complete tool call."""
        call = next((c for c in tool_calls if c.get("name") == name), None)
        if not call:
            return False
        params = call.get("params") or {}
        return all(params.get(k) for k in required_params)

    def _fallback(name: str, params: dict, reason: str) -> None:
        """Replace router output with one deterministic fallback tool call."""
        nonlocal tool_calls
        tool_calls = [{"name": name, "params": params}]
        if debug:
            debug_info.append(f"**Fallback:** {reason} -> {name}({params})")

    def _fallback_many(calls: list[dict], reason: str) -> None:
        """Replace router output with multiple deterministic fallback calls."""
        nonlocal tool_calls
        tool_calls = calls
        if debug:
            debug_info.append(f"**Fallback:** {reason} -> {calls}")

    route_triggers = {
        "fastest route",
        "quickest route",
        "shortest route",
        "cheapest route",
        "best route",
        "how to get",
        "directions from",
        "route from",
        "route to",
        "get from",
        "travel from",
        "way from",
        "path from",
        "最快路線",
        "最短路線",
        "最便宜路線",
        "最便宜",
        "怎麼去",
        "如何前往",
        "路線規劃",
        "路線查詢",
        "怎麼走",
        "如何去",
        "如何搭",
        "怎麼搭",
    }
    fare_triggers = {
        "fare",
        "price",
        "cost",
        "ticket price",
        "多少錢",
        "票價",
        "價格",
        "費用",
    }
    booking_precheck_triggers = {
        "book",
        "booking",
        "reserve",
        "reservation",
        "buy a ticket",
        "ticket",
        "訂票",
        "訂一張票",
        "買票",
        "預訂",
    }
    travel_date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", augmented_message)
    # For unconfirmed rail booking requests, gather availability, prices, and seats first.
    is_booking_precheck = (
        any(kw in lower for kw in booking_precheck_triggers)
        and not _user_confirmed(history + [{"role": "user", "content": user_message}])
        and len(station_ids) >= 2
        and schedule_ids
        and travel_date_match
        and schedule_ids[0].upper().startswith("NR_SCH")
    )
    if is_booking_precheck:
        # Build a multi-tool pre-check so the user can confirm with complete details.
        schedule_id = schedule_ids[0].upper()
        origin_id = station_ids[0].upper()
        destination_id = station_ids[1].upper()
        travel_date = travel_date_match.group(0)
        fare_class = "first" if "first" in lower else "standard"
        _fallback_many(
            [
                {
                    "name": "check_national_rail_availability",
                    "params": {
                        "origin_id": origin_id,
                        "destination_id": destination_id,
                        "travel_date": travel_date,
                    },
                },
                {
                    "name": "get_national_rail_schedule_fares",
                    "params": {"schedule_id": schedule_id},
                },
                {
                    "name": "get_available_seats",
                    "params": {
                        "schedule_id": schedule_id,
                        "travel_date": travel_date,
                        "fare_class": fare_class,
                    },
                },
            ],
            "booking pre-check needs route/date availability, fare, and seats",
        )

    if (
        schedule_ids
        and any(kw in lower for kw in fare_triggers)
        and not _tool_selected("get_national_rail_schedule_fares", "schedule_id")
        and not is_booking_precheck
    ):
        # If the router missed a rail fare question, recover from the schedule ID.
        schedule_id = schedule_ids[0].upper()
        if schedule_id.startswith("NR_SCH"):
            _fallback(
                "get_national_rail_schedule_fares",
                {"schedule_id": schedule_id},
                "national rail schedule fare query",
            )

    is_route = (
        any(kw in lower for kw in route_triggers)
        or (two_stations and "route" in lower)
        or (two_stations and "路線" in lower)
    )
    if is_route and two_stations:
        # Repair router mistakes caused by duplicated station IDs after name injection.
        for call in tool_calls:
            if call.get("name") != "find_route":
                continue
            params = call.setdefault("params", {})
            origin_id = str(params.get("origin_id", "")).upper()
            destination_id = str(params.get("destination_id", "")).upper()
            if origin_id == destination_id and station_ids[0] == origin_id:
                params["destination_id"] = station_ids[1]
                if debug:
                    debug_info.append(
                        f"**Route repair:** destination changed from {destination_id} to {station_ids[1]}"
                    )

    if is_route and two_stations and not _tool_selected("find_route", "origin_id", "destination_id"):
        # Route fallback catches common "how do I get from A to B" phrasing.
        optimise_by = "cost" if any(
            kw in lower for kw in ["cheap", "cheapest", "lowest cost", "最便宜", "最低票價"]
        ) else "time"
        _fallback(
            "find_route",
            {
                "origin_id": station_ids[0].upper(),
                "destination_id": station_ids[1].upper(),
                "optimise_by": optimise_by,
            },
            "route query",
        )

    elif not tool_calls and two_stations:
        # Availability fallback catches schedule/timetable questions with two station IDs.
        availability_triggers = {
            "train",
            "trains",
            "service",
            "services",
            "run from",
            "runs from",
            "schedule",
            "timetable",
            "available",
            "availability",
            "班次",
            "時刻表",
            "列車",
            "有沒有車",
            "幾點有車",
            "查車",
        }
        if any(kw in lower for kw in availability_triggers):
            origin_id = station_ids[0].upper()
            destination_id = station_ids[1].upper()
            travel_date = next(
                (w for w in lower.split() if re.match(r"\d{4}-\d{2}-\d{2}", w)),
                None,
            )
            params = {"origin_id": origin_id, "destination_id": destination_id}
            if travel_date:
                params["travel_date"] = travel_date
            tool = "check_national_rail_availability" if origin_id.startswith("NR") else "check_metro_availability"
            _fallback(tool, params, "availability query")

    if current_user_email and not tool_calls:
        # Logged-in personal-history questions should read bookings without extra params.
        personal_triggers = {
            "my booking",
            "my ticket",
            "my trip",
            "my journey",
            "my history",
            "my reservation",
            "show booking",
            "view booking",
            "check booking",
            "list booking",
            "show my",
            "view my",
            "我的訂票",
            "我的票",
            "我的行程",
            "訂票紀錄",
            "查詢訂票",
            "我訂的",
            "我的車票",
        }
        if any(kw in lower for kw in personal_triggers):
            _fallback("get_user_bookings", {}, "personal booking query")

    if current_user_email and not tool_calls:
        # Logged-in profile/account questions read the user's registered profile.
        profile_triggers = {
            "my account",
            "my profile",
            "my info",
            "account details",
            "我的帳號",
            "我的資料",
            "帳號資訊",
            "個人資料",
        }
        if any(kw in lower for kw in profile_triggers):
            _fallback("get_user_profile", {}, "profile query")

    if not tool_calls:
        # Policy questions are answered from vector-searched policy documents.
        policy_triggers = {
            "refund",
            "policy",
            "compensation",
            "luggage",
            "bicycle",
            "pet",
            "退款",
            "補償",
            "政策",
            "行李",
            "寵物",
            "腳踏車",
            "規定",
        }
        if any(kw in lower for kw in policy_triggers):
            _fallback("search_policy", {"query": user_message}, "policy query")

    if any(c.get("name") == "make_booking" for c in tool_calls):
        # Never allow booking writes unless the latest user message confirms it.
        if not _user_confirmed(history + [{"role": "user", "content": user_message}]):
            tool_calls = []
            if debug:
                debug_info.append("**Booking gate:** make_booking blocked; no confirmation detected.")

    tool_results = []
    for call in tool_calls:
        # Execute each selected tool and keep both raw JSON and LLM-facing summary.
        tool_name = call.get("name", "")
        params = call.get("params") or call.get("parameters", {})

        # Empty-string parameters usually mean the router guessed, so skip them safely.
        if any(v == "" for v in params.values()):
            if debug:
                debug_info.append(f"**Skipped** `{tool_name}` - empty params: {params}")
            continue

        if debug:
            debug_info.append(f"**Calling:** `{tool_name}({params})`")

        result_json = _execute_tool(tool_name, params, current_user_email)
        summary = _summarise_result(tool_name, result_json)

        if debug:
            debug_info.append(
                f"**Result (raw):** ```json\n{result_json[:300]}\n```\n"
                f"**Summary sent to LLM:** {summary}"
            )

        tool_results.append(
            {
                "tool": tool_name,
                "params": params,
                "result": result_json,
                "summary": summary,
            }
        )

    db_keywords = {
        "booking",
        "ticket",
        "schedule",
        "fare",
        "route",
        "seat",
        "train",
        "metro",
        "journey",
        "trip",
        "history",
        "reservation",
        "訂票",
        "班次",
        "票價",
        "路線",
        "座位",
        "捷運",
        "列車",
    }
    data_block = ""
    if tool_results:
        # Feed database results back to the LLM as the only source of truth.
        data_block = "\n\n".join(
            f"[{tr['tool']}]\n{_normalise_result(tr['tool'], tr['result'])}"
            for tr in tool_results
        )
        if debug:
            debug_info.append(f"**Data (normalised):**\n{data_block}")
        content = (
            f"DATA FROM TRANSITFLOW DATABASE:\n{data_block}"
            f"\n\nUser asks: {user_message}"
            f"\n\nAnswer using only the data above. Use emojis and clear formatting:"
        )
    elif any(kw in user_message.lower() for kw in db_keywords):
        # For database-like questions with no data, make the model avoid hallucination.
        content = (
            f"User asks: {user_message}\n\n"
            "IMPORTANT: No data was retrieved from the TransitFlow database for this query. "
            "Apologise politely in the user's language and suggest what they can try instead. "
            "Do NOT invent any bookings, fares, schedules, seat numbers, or travel times."
        )
    else:
        # Non-database chat can go directly to the LLM without tool data.
        content = user_message

    final_messages = history + [{"role": "user", "content": content}]
    try:
        # Final generation turns tool data into a friendly user-facing answer.
        answer = llm.chat(messages=final_messages, system_prompt=contextual_prompt)
    except ConnectionError as exc:
        # If the LLM is down but tools worked, still show the retrieved data.
        if tool_results:
            answer = f"目前 Ollama 沒有啟動；先直接提供資料庫查詢結果：\n\n{data_block}"
        else:
            answer = (
                "目前 Ollama 沒有啟動，所以我無法產生 AI 回覆。"
                "請啟動 Ollama，或在右側切換到 Gemini 後再試一次。"
            )
        if debug:
            debug_info.append(f"**LLM unavailable:** {exc}")

    # Persist the new user/assistant turn for the UI chat state.
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]

    if debug:
        return answer, updated_history, "\n\n".join(debug_info)
    return answer, updated_history

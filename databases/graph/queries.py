"""
TransitFlow Neo4j Graph Database Layer.

This module handles route and disruption queries over the dual metro / national
rail graph seeded by skeleton/seed_neo4j.py.
"""

from __future__ import annotations

import heapq
from typing import Optional

from neo4j import GraphDatabase

from skeleton.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def _driver():
    """Return a Neo4j driver. Caller is responsible for closing."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def example_count_nodes() -> int:
    """Example: count all nodes currently in the graph."""
    with _driver() as driver:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS total")
            return result.single()["total"]


def _relationship_type(rel) -> str:
    rel_type = getattr(rel, "type", None)
    if rel_type:
        return rel_type
    return type(rel).__name__


def _station_dict(node) -> dict:
    return {
        "station_id": node.get("station_id"),
        "name": node.get("name"),
        "network": node.get("network"),
        "lines": list(node.get("lines") or []),
    }


def _leg_dict(start_node, rel, end_node) -> dict:
    return {
        "from_station_id": start_node.get("station_id"),
        "from_name": start_node.get("name"),
        "to_station_id": end_node.get("station_id"),
        "to_name": end_node.get("name"),
        "relationship": _relationship_type(rel),
        "line": rel.get("line"),
        "travel_time_min": rel.get("travel_time_min", 0),
        "fare_standard_usd": float(rel.get("fare_standard_usd", 0) or 0),
        "fare_first_usd": float(rel.get("fare_first_usd", 0) or 0),
    }


def _network_relationships(network: str) -> str:
    network = (network or "auto").lower()
    if network == "metro":
        return "METRO_LINK"
    if network == "rail":
        return "RAIL_LINK"
    return "METRO_LINK|RAIL_LINK|INTERCHANGE_TO"


def _infer_network(origin_id: str, destination_id: str, network: str) -> str:
    if network and network.lower() != "auto":
        return network.lower()
    if origin_id.startswith("MS") and destination_id.startswith("MS"):
        return "metro"
    if origin_id.startswith("NR") and destination_id.startswith("NR"):
        return "rail"
    return "auto"


def _not_found(origin_id: str, destination_id: str) -> dict:
    return {
        "found": False,
        "origin_id": origin_id,
        "destination_id": destination_id,
        "total_time_min": None,
        "total_fare_usd": None,
        "path": [],
        "stations": [],
        "legs": [],
    }


def _load_graph() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    cypher = """
        MATCH (station:Station)
        OPTIONAL MATCH (station)-[rel]->(connected:Station)
        RETURN
            station,
            collect({
                rel: rel,
                connected: connected
            }) AS raw_edges
    """
    stations: dict[str, dict] = {}
    edges: dict[str, list[dict]] = {}
    with _driver() as driver:
        with driver.session() as session:
            for record in session.run(cypher):
                station = record["station"]
                station_id = station.get("station_id")
                stations[station_id] = _station_dict(station)
                edges[station_id] = []
                for raw_edge in record["raw_edges"]:
                    rel = raw_edge.get("rel")
                    connected = raw_edge.get("connected")
                    if rel is None or connected is None:
                        continue
                    edges[station_id].append(
                        {
                            "to": connected.get("station_id"),
                            "to_station": _station_dict(connected),
                            "relationship": _relationship_type(rel),
                            "line": rel.get("line"),
                            "travel_time_min": rel.get("travel_time_min", 0),
                            "fare_standard_usd": float(rel.get("fare_standard_usd", 0) or 0),
                            "fare_first_usd": float(rel.get("fare_first_usd", 0) or 0),
                        }
                    )
    return stations, edges


def _edge_allowed(edge: dict, relationship_filter: str) -> bool:
    allowed = set(relationship_filter.split("|"))
    return edge["relationship"] in allowed


def _edge_cost(edge: dict, optimise_by: str, fare_property: str) -> float:
    if optimise_by == "fare":
        return float(edge.get(fare_property, 0) or 0)
    return float(edge.get("travel_time_min", 0) or 0)


def _build_route(
    origin_id: str,
    destination_id: str,
    stations: dict[str, dict],
    path_edges: list[tuple[str, dict]],
) -> dict:
    station_ids = [origin_id] + [edge["to"] for _, edge in path_edges]
    station_path = [stations[station_id] for station_id in station_ids]
    legs = []
    for from_station_id, edge in path_edges:
        start_station = stations[from_station_id]
        end_station = stations[edge["to"]]
        legs.append(
            {
                "from_station_id": from_station_id,
                "from_name": start_station["name"],
                "to_station_id": edge["to"],
                "to_name": end_station["name"],
                "relationship": edge["relationship"],
                "line": edge["line"],
                "travel_time_min": edge["travel_time_min"],
                "fare_standard_usd": edge["fare_standard_usd"],
                "fare_first_usd": edge["fare_first_usd"],
            }
        )
    return {
        "found": True,
        "origin_id": origin_id,
        "destination_id": destination_id,
        "total_time_min": sum(leg["travel_time_min"] for leg in legs),
        "total_fare_usd": round(sum(leg["fare_standard_usd"] for leg in legs), 2),
        "path": station_path,
        "stations": station_path,
        "legs": legs,
    }


def _route_query(
    origin_id: str,
    destination_id: str,
    network: str = "auto",
    optimise_by: str = "time",
    fare_property: str = "fare_standard_usd",
    avoid_station_id: Optional[str] = None,
    limit: int = 1,
) -> list[dict]:
    relationship_filter = _network_relationships(
        _infer_network(origin_id, destination_id, network)
    )
    if fare_property not in {"fare_standard_usd", "fare_first_usd"}:
        fare_property = "fare_standard_usd"

    stations, edges = _load_graph()
    if origin_id not in stations or destination_id not in stations:
        return []

    max_routes = max(1, int(limit or 1))
    queue: list[tuple[float, int, str, list[tuple[str, dict]], set[str]]] = [
        (0.0, 0, origin_id, [], {origin_id})
    ]
    routes: list[dict] = []
    tie_breaker = 1
    while queue and len(routes) < max_routes:
        _, _, current_id, path_edges, visited = heapq.heappop(queue)
        if current_id == destination_id:
            route = _build_route(origin_id, destination_id, stations, path_edges)
            if optimise_by == "fare":
                route["total_fare_usd"] = round(
                    sum(edge.get(fare_property, 0) for _, edge in path_edges), 2
                )
            routes.append(route)
            continue

        for edge in edges.get(current_id, []):
            next_id = edge["to"]
            if next_id in visited:
                continue
            if avoid_station_id and next_id == avoid_station_id and next_id != destination_id:
                continue
            if not _edge_allowed(edge, relationship_filter):
                continue
            next_path = path_edges + [(current_id, edge)]
            next_visited = set(visited)
            next_visited.add(next_id)
            cost = sum(_edge_cost(item, optimise_by, fare_property) for _, item in next_path)
            heapq.heappush(queue, (cost, tie_breaker, next_id, next_path, next_visited))
            tie_breaker += 1
    return routes


def query_shortest_route(
    origin_id: str,
    destination_id: str,
    network: str = "auto",
) -> dict:
    """
    Find the fastest path between two stations, minimising total travel time.
    """
    routes = _route_query(
        origin_id=origin_id,
        destination_id=destination_id,
        network=network,
        optimise_by="time",
        limit=1,
    )
    return routes[0] if routes else _not_found(origin_id, destination_id)


def query_cheapest_route(
    origin_id: str,
    destination_id: str,
    network: str = "auto",
    fare_class: str = "standard",
) -> dict:
    """
    Find the cheapest path between two stations, minimising approximate fare.
    """
    fare_property = "fare_first_usd" if fare_class == "first" else "fare_standard_usd"
    routes = _route_query(
        origin_id=origin_id,
        destination_id=destination_id,
        network=network,
        optimise_by="fare",
        fare_property=fare_property,
        limit=1,
    )
    if not routes:
        return _not_found(origin_id, destination_id)
    routes[0]["fare_class"] = fare_class
    return routes[0]


def query_alternative_routes(
    origin_id: str,
    destination_id: str,
    avoid_station_id: str,
    network: str = "auto",
    max_routes: int = 3,
) -> list[list[dict]]:
    """
    Find route legs for paths that avoid a specific intermediate station.
    """
    routes = _route_query(
        origin_id=origin_id,
        destination_id=destination_id,
        network=network,
        optimise_by="time",
        avoid_station_id=avoid_station_id,
        limit=max_routes,
    )
    return [route["legs"] for route in routes]


def query_interchange_path(origin_id: str, destination_id: str) -> dict:
    """
    Find a path between metro and national rail, crossing via interchange links.
    """
    routes = _route_query(
        origin_id=origin_id,
        destination_id=destination_id,
        network="auto",
        optimise_by="time",
        limit=1,
    )
    if not routes:
        return _not_found(origin_id, destination_id)

    route = routes[0]
    route["interchange_points"] = [
        {
            "from_station_id": leg["from_station_id"],
            "from_name": leg["from_name"],
            "to_station_id": leg["to_station_id"],
            "to_name": leg["to_name"],
        }
        for leg in route["legs"]
        if leg["relationship"] == "INTERCHANGE_TO"
    ]
    return route


def query_delay_ripple(delayed_station_id: str, hops: int = 2) -> list[dict]:
    """
    Find all stations within N hops of a delayed or disrupted station.
    """
    hops = max(1, min(int(hops or 2), 6))
    stations, edges = _load_graph()
    if delayed_station_id not in stations:
        return []

    queue: list[tuple[str, int, set[str]]] = [(delayed_station_id, 0, set())]
    best: dict[str, tuple[int, set[str]]] = {}
    seen_depth: dict[str, int] = {delayed_station_id: 0}

    while queue:
        current_id, depth, lines = queue.pop(0)
        if depth >= hops:
            continue
        for edge in edges.get(current_id, []):
            next_id = edge["to"]
            next_depth = depth + 1
            next_lines = set(lines)
            if edge.get("line"):
                next_lines.add(edge["line"])
            if next_id == delayed_station_id:
                continue
            previous_depth = seen_depth.get(next_id)
            if previous_depth is not None and previous_depth < next_depth:
                continue
            seen_depth[next_id] = next_depth
            current_best = best.get(next_id)
            if current_best is None or next_depth < current_best[0]:
                best[next_id] = (next_depth, next_lines)
            else:
                current_best[1].update(next_lines)
            queue.append((next_id, next_depth, next_lines))

    return [
        {
            "station_id": station_id,
            "name": stations[station_id]["name"],
            "network": stations[station_id]["network"],
            "hops_away": depth,
            "lines_affected": sorted(lines),
        }
        for station_id, (depth, lines) in sorted(
            best.items(), key=lambda item: (item[1][0], item[0])
        )
    ]


def query_station_connections(station_id: str) -> list[dict]:
    """
    List all direct outbound connections from a given station.
    """
    cypher = """
        MATCH (station:Station {station_id: $station_id})-[rel]->(connected:Station)
        RETURN
            connected.station_id AS station_id,
            connected.name AS name,
            connected.network AS network,
            type(rel) AS relationship,
            rel.line AS line,
            rel.travel_time_min AS travel_time_min,
            rel.fare_standard_usd AS fare_standard_usd,
            rel.fare_first_usd AS fare_first_usd
        ORDER BY relationship, line, station_id
    """
    with _driver() as driver:
        with driver.session() as session:
            return [dict(record) for record in session.run(cypher, station_id=station_id)]

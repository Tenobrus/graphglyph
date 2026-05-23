#!/usr/bin/env python3
"""
Lattice Bloom graph encoder.

Encodes UTF-8 text into a dense blue/orange graph and decodes it back from the
generated SVG or JSON graph file. The plaintext is not stored in metadata; it is
recovered from weighted data edges in the graph topology.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import html
import itertools
import json
import math
import random
import struct
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WINDOW_CHOICES = ("seeded", "norm", "bounded-norm", "box")
SCHEME = "lattice-bloom-v2"
MAGIC = b"LBM2"
VERSION = 2
FLAGS = 0
FLAG_ZLIB = 1
SEED_LEN = 8
HEADER_LEN = 4 + 1 + 1 + SEED_LEN + 4
CRC_LEN = 4
HEADER_CELLS = HEADER_LEN * 2
PERMUTE_XOR = 0xC0DEC0DEC0DEC0DE
LAYOUT_XOR = 0x9E3779B97F4A7C15

DATA_WEAK_WEIGHT = 0.30
DATA_STRONG_WEIGHT = 0.62
VARIANT_XOR = 0x6A09E667F3BCC909


@dataclass(frozen=True)
class Node:
    id: str
    x: float
    y: float


@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    weight: float
    data_cell: int | None = None
    data_slot: int | None = None
    data_bit: int | None = None


@dataclass(frozen=True)
class Graph:
    width: float
    height: float
    point_cols: int
    point_rows: int
    data_cells: int
    nodes: list[Node]
    edges: list[Edge]


@dataclass(frozen=True)
class Header:
    flags: int
    seed: int
    payload_len: int


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def make_parts(text: str) -> tuple[bytes, bytes, int, str]:
    normalized = normalize_text(text)
    payload = normalized.encode("utf-8")
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("text is too large for lattice-bloom-v2")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:SEED_LEN], "big")
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    stored_payload = payload
    flags = FLAGS
    compressed = zlib.compress(payload, level=9)
    if len(compressed) + 16 < len(payload):
        stored_payload = compressed
        flags |= FLAG_ZLIB
    header = MAGIC + bytes([VERSION, flags]) + seed.to_bytes(SEED_LEN, "big") + struct.pack(">I", len(stored_payload))
    tail = stored_payload + struct.pack(">I", crc)
    return header, tail, seed, normalized


def parse_header(data: bytes) -> Header:
    if len(data) < HEADER_LEN:
        raise ValueError("graph does not contain enough header bytes")
    if data[:4] != MAGIC:
        raise ValueError("graph does not start with a lattice-bloom packet")
    version = data[4]
    if version != VERSION:
        raise ValueError(f"unsupported packet version {version}")
    flags = data[5]
    if flags & ~FLAG_ZLIB:
        raise ValueError(f"unsupported packet flags {flags}")
    seed = int.from_bytes(data[6 : 6 + SEED_LEN], "big")
    payload_len = struct.unpack(">I", data[6 + SEED_LEN : HEADER_LEN])[0]
    return Header(flags=flags, seed=seed, payload_len=payload_len)


def parse_tail(header: Header, data: bytes) -> str:
    total_len = header.payload_len + CRC_LEN
    if len(data) < total_len:
        raise ValueError("graph ended before the encoded payload was complete")
    stored_payload = data[: header.payload_len]
    if header.flags & FLAG_ZLIB:
        payload = zlib.decompress(stored_payload)
    else:
        payload = stored_payload
    expected_crc = struct.unpack(">I", data[header.payload_len : total_len])[0]
    actual_crc = binascii.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError("CRC check failed; this graph is damaged or not from this encoder")
    return payload.decode("utf-8")


def bytes_to_nibbles(data: bytes) -> list[int]:
    nibbles: list[int] = []
    for byte in data:
        nibbles.append((byte >> 4) & 0x0F)
        nibbles.append(byte & 0x0F)
    return nibbles


def nibbles_to_bytes(nibbles: Iterable[int]) -> bytes:
    values = list(nibbles)
    if len(values) % 2:
        values.append(0)
    out = bytearray()
    for i in range(0, len(values), 2):
        out.append(((values[i] & 0x0F) << 4) | (values[i + 1] & 0x0F))
    return bytes(out)


def normalized_edge(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def add_visual_edge(edges: dict[tuple[str, str], float], u: str, v: str, weight: float) -> None:
    if u == v:
        return
    if weight <= 0:
        return
    key = normalized_edge(u, v)
    edges[key] = max(edges.get(key, 0.0), weight)


def project_coeffs(coeffs: tuple[int, ...], basis: list[complex]) -> complex:
    return sum(value * direction for value, direction in zip(coeffs, basis))


def stable_unit(seed: int, *parts: object) -> float:
    digest = hashlib.sha256(f"{seed}:{parts!r}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def exact_rho_basis() -> list[complex]:
    rho = complex(-0.5, math.sqrt(3.0) / 2.0)
    return [1.0 + 0.0j, 1j, rho, 1j * rho]


def make_projection_basis(seed: int, variant_strength: float) -> list[complex]:
    strength = max(0.0, min(1.0, variant_strength))
    if strength <= 1e-9:
        return exact_rho_basis()

    rng = random.Random(seed ^ VARIANT_XOR)
    families = [3, 12]
    family_selector = (stable_unit(seed, "basis-family") + stable_unit(seed, "basis-family-alt")) / 2.0
    family = families[int(family_selector * len(families)) % len(families)]
    if strength < 0.35:
        family = 3
    theta = math.tau / family
    alpha = complex(math.cos(theta), math.sin(theta))
    rotation = rng.uniform(0.0, math.tau) * strength
    rotator = complex(math.cos(rotation), math.sin(rotation))
    return [rotator, 1j * rotator, alpha * rotator, 1j * alpha * rotator]


def make_shadow_basis(seed: int, dimension: int) -> list[complex]:
    rng = random.Random(seed ^ 0xBB67AE8584CAA73B)
    angles = [rng.uniform(0.0, math.tau) for _ in range(dimension)]
    return [complex(math.cos(angle), math.sin(angle)) for angle in angles]


def choose_coefficients(
    range_limit: int,
    seed: int,
    variant_strength: float,
    basis: list[complex],
) -> list[tuple[int, ...]]:
    strength = max(0.0, min(1.0, variant_strength))
    dimension = len(basis)
    if strength <= 1e-9 and dimension == 4:
        return list(itertools.product(range(-range_limit, range_limit + 1), repeat=dimension))

    rng = random.Random(seed ^ VARIANT_XOR ^ (range_limit * 0x9E3779B1))
    search_limit = range_limit + 1 + int(strength > 0.62)
    base_target_count = (2 * range_limit + 1) ** 4
    if strength <= 1e-9:
        target_count = base_target_count
    else:
        target_factor = 0.62 + 0.95 * stable_unit(seed, "point-count")
        target_count = max(HEADER_CELLS + 96, int(base_target_count * target_factor))
    radius1 = (range_limit + 0.8) * math.sqrt(max(1.0, dimension / 4.0)) * rng.uniform(0.80, 1.22)
    radius2 = (range_limit + 0.8) * math.sqrt(max(1.0, dimension / 4.0)) * rng.uniform(0.78, 1.28)
    center_scale = range_limit * (0.30 + 0.70 * strength)
    center1 = complex(rng.uniform(-center_scale, center_scale), rng.uniform(-center_scale, center_scale))
    center2 = complex(rng.uniform(-center_scale, center_scale), rng.uniform(-center_scale, center_scale))
    center3 = complex(rng.uniform(-center_scale, center_scale), rng.uniform(-center_scale, center_scale))
    coeff_center = [rng.uniform(-range_limit * 0.55 * strength, range_limit * 0.55 * strength) for _ in range(dimension)]
    shadow_basis = make_shadow_basis(seed, dimension)
    wave = [rng.uniform(-math.pi, math.pi) for _ in range(dimension)]
    window_mode = stable_unit(seed, "window-mode")

    scored: list[tuple[float, float, tuple[int, ...]]] = []
    for coeffs in itertools.product(range(-search_limit, search_limit + 1), repeat=dimension):
        z1 = project_coeffs(coeffs, basis)
        z2 = project_coeffs(coeffs, shadow_basis)
        box_score = max(abs(value) / max(1.0, float(range_limit)) for value in coeffs)
        shifted_box_score = max(abs(value - coeff_center[index]) / max(1.0, float(range_limit)) for index, value in enumerate(coeffs))
        disc_score = max(abs(z1 - center1) / radius1, abs(z2 - center2) / radius2)
        annulus_score = abs(abs(z1 - center1) - radius1 * 0.68) / (radius1 * 0.30) + 0.28 * abs(z2 - center2) / radius2
        lobe_score = min(abs(z1 - center1), abs(z1 - center3)) / radius1 + 0.34 * abs(z2 - center2) / radius2
        if window_mode < 0.34:
            window_score = disc_score
        elif window_mode < 0.67:
            window_score = 0.72 * disc_score + 0.28 * annulus_score
        else:
            window_score = 0.62 * disc_score + 0.38 * lobe_score
        ripple_arg = sum((value + 0.5) * wave[index] for index, value in enumerate(coeffs))
        ripple = math.sin(ripple_arg) * 0.045 * strength
        score = (1.0 - strength) * box_score + strength * (0.76 * window_score + 0.24 * shifted_box_score) + ripple
        scored.append((score, stable_unit(seed, "coeff", coeffs), coeffs))

    scored.sort()
    return [coeffs for _score, _tie_breaker, coeffs in scored[:target_count]]


def coefficient_box(range_limit: int, dimension: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(-range_limit, range_limit + 1), repeat=dimension))


def choose_norm_coefficients(
    range_limit: int,
    basis: list[complex],
    norm_radius: float,
) -> list[tuple[int, ...]]:
    if norm_radius <= 0:
        raise ValueError("--norm-radius must be positive")
    return [coeffs for coeffs in coefficient_box(range_limit, len(basis)) if abs(project_coeffs(coeffs, basis)) < norm_radius]


def choose_window_coefficients(
    window: str,
    range_limit: int,
    seed: int,
    variant_strength: float,
    norm_radius: float,
) -> tuple[list[complex], list[tuple[int, ...]], float]:
    if window == "seeded":
        strength = max(0.0, min(1.0, variant_strength))
        basis = make_projection_basis(seed, strength)
        return basis, choose_coefficients(range_limit, seed, strength, basis), strength
    if window in {"norm", "bounded-norm"}:
        basis = exact_rho_basis()
        return basis, choose_norm_coefficients(range_limit, basis, norm_radius), 0.0
    if window == "box":
        basis = exact_rho_basis()
        return basis, coefficient_box(range_limit, len(basis)), 0.0
    raise ValueError(f"unknown coefficient window {window!r}")


def unit_step_vectors(basis: list[complex]) -> list[tuple[int, ...]]:
    dimension = len(basis)
    vectors: list[tuple[int, ...]] = []
    for delta in itertools.product(range(-1, 2), repeat=dimension):
        if all(value == 0 for value in delta):
            continue
        z = project_coeffs(delta, basis)
        if abs(abs(z) - 1.0) < 1e-9:
            vectors.append(delta)
    return vectors


def make_unit_distance_substrate(
    range_limit: int,
    unit_px: float,
    seed: int,
    window: str,
    variant_strength: float,
    norm_radius: float,
) -> tuple[list[Node], set[int], dict[tuple[str, str], float], dict[int, list[int]]]:
    basis, coefficients, visual_strength = choose_window_coefficients(window, range_limit, seed, variant_strength, norm_radius)
    raw_points: list[tuple[float, float, tuple[int, ...]]] = []
    seen: set[tuple[float, float]] = set()
    for coeffs in coefficients:
        z = project_coeffs(coeffs, basis)
        x = z.real
        y = z.imag
        key = (round(x, 12), round(y, 12))
        if key in seen:
            continue
        seen.add(key)
        raw_points.append((x, y, coeffs))

    raw_points.sort(key=lambda item: (item[1], item[0], item[2]))
    min_x = min(x for x, _y, _coeffs in raw_points)
    max_y = max(y for _x, y, _coeffs in raw_points)
    margin = unit_px * 0.10
    nodes = [
        Node(f"p{index}", (x - min_x) * unit_px + margin, (max_y - y) * unit_px + margin)
        for index, (x, y, _coeffs) in enumerate(raw_points)
    ]

    visual_edges: dict[tuple[str, str], float] = {}
    incident: dict[int, list[int]] = {index: [] for index in range(len(nodes))}
    coeff_to_index = {coeffs: index for index, (_x, _y, coeffs) in enumerate(raw_points)}
    unit_vectors = unit_step_vectors(basis)
    for i, (_x, _y, coeffs) in enumerate(raw_points):
        for delta in unit_vectors:
            target = tuple(value + shift for value, shift in zip(coeffs, delta))
            j = coeff_to_index.get(target)
            if j is None or j <= i:
                continue
            edge_noise = stable_unit(seed, "edge", coeffs, target) - 0.5
            add_visual_edge(visual_edges, nodes[i].id, nodes[j].id, 0.54 + edge_noise * 0.22 * visual_strength)
            incident[i].append(j)
            incident[j].append(i)

    for center_index, neighbors in incident.items():
        center = nodes[center_index]
        neighbors.sort(key=lambda index: math.atan2(nodes[index].y - center.y, nodes[index].x - center.x))

    preferred = {index for index, neighbors in incident.items() if len(neighbors) >= 8}
    return nodes, preferred, visual_edges, incident


def choose_data_centers(
    nodes: list[Node],
    point_cols: int,
    point_rows: int,
    data_cell_count: int,
    seed: int,
    preferred: set[int] | None = None,
) -> list[int]:
    interior: list[int] = []
    if point_cols > 0 and point_rows > 0:
        for index in range(len(nodes)):
            row, col = divmod(index, point_cols)
            if 2 <= row < point_rows - 2 and 2 <= col < point_cols - 2:
                interior.append(index)
    else:
        min_x = min(node.x for node in nodes)
        max_x = max(node.x for node in nodes)
        min_y = min(node.y for node in nodes)
        max_y = max(node.y for node in nodes)
        pad_x = (max_x - min_x) * 0.055
        pad_y = (max_y - min_y) * 0.055
        for index, node in enumerate(nodes):
            if min_x + pad_x <= node.x <= max_x - pad_x and min_y + pad_y <= node.y <= max_y - pad_y:
                interior.append(index)
    if len(interior) < data_cell_count:
        interior = list(range(len(nodes)))
    rng = random.Random(seed ^ (LAYOUT_XOR >> 1))
    rng.shuffle(interior)
    if preferred:
        preferred_order = [index for index in interior if index in preferred]
        other_order = [index for index in interior if index not in preferred]
        interior = preferred_order + other_order
    if len(interior) < data_cell_count:
        raise ValueError("not enough graph points to place data cells")
    return interior[:data_cell_count]


def build_graph(
    text: str,
    *,
    cell_size: float = 52.0,
    min_cells: int = 120,
    padding: int = 24,
    unit_range: int = 2,
    window: str = "seeded",
    norm_radius: float = 4.0,
    variant_strength: float = 0.75,
) -> Graph:
    header, tail, seed, _normalized = make_parts(text)
    pad_rng = random.Random(seed ^ 0x517CC1B727220A95)

    header_nibbles = bytes_to_nibbles(header)
    tail_nibbles = bytes_to_nibbles(tail)
    wanted_cells = max(len(header_nibbles) + len(tail_nibbles) + padding, min_cells)
    data_cell_count = wanted_cells

    nibbles = [pad_rng.randrange(16) for _ in range(data_cell_count)]
    nibbles[: len(header_nibbles)] = header_nibbles
    data_cells = list(range(len(header_nibbles), data_cell_count))
    perm_rng = random.Random(seed ^ PERMUTE_XOR)
    perm_rng.shuffle(data_cells)
    for nibble, cell in zip(tail_nibbles, data_cells):
        nibbles[cell] = nibble

    effective_unit_range = unit_range
    while True:
        nodes, round_motif_centers, style_visual_edges, unit_incident = make_unit_distance_substrate(
            effective_unit_range,
            cell_size,
            seed,
            window,
            variant_strength,
            norm_radius,
        )
        if len(round_motif_centers) >= data_cell_count:
            break
        effective_unit_range += 1
    point_cols = 0
    point_rows = 0
    center_indices = choose_data_centers(
        nodes,
        point_cols,
        point_rows,
        data_cell_count,
        seed,
        preferred=round_motif_centers,
    )

    visual_edges: dict[tuple[str, str], float] = dict(style_visual_edges)
    data_edges: list[Edge] = []

    for cell, center_index in enumerate(center_indices):
        center_id = nodes[center_index].id
        neighbor_indices = unit_incident[center_index]
        if len(neighbor_indices) < 8:
            raise ValueError("selected data center lacks enough unit-distance neighbors")
        nibble = nibbles[cell]
        for slot in range(4):
            bit = (nibble >> (3 - slot)) & 1
            pair = (
                neighbor_indices[(2 * slot) % len(neighbor_indices)],
                neighbor_indices[(2 * slot + 1) % len(neighbor_indices)],
            )
            for candidate_bit, neighbor_index in enumerate(pair):
                weight = DATA_STRONG_WEIGHT if candidate_bit == bit else DATA_WEAK_WEIGHT
                data_edges.append(
                    Edge(
                        center_id,
                        nodes[neighbor_index].id,
                        weight,
                        data_cell=cell,
                        data_slot=slot,
                        data_bit=candidate_bit,
                    )
                )

    visual_edge_list = [Edge(u, v, weight) for (u, v), weight in sorted(visual_edges.items())]
    margin = cell_size * 0.08
    min_x = min(node.x for node in nodes)
    min_y = min(node.y for node in nodes)
    nodes = [Node(node.id, node.x - min_x + margin, node.y - min_y + margin) for node in nodes]
    width = max(node.x for node in nodes) + margin
    height = max(node.y for node in nodes) + margin
    return Graph(width, height, point_cols, point_rows, data_cell_count, nodes, visual_edge_list + data_edges)


def graph_to_json(graph: Graph) -> dict:
    edges = []
    for edge in graph.edges:
        item = {"u": edge.u, "v": edge.v, "weight": round(edge.weight, 4)}
        if edge.data_cell is not None:
            item.update({"data_cell": edge.data_cell, "data_slot": edge.data_slot, "data_bit": edge.data_bit})
        edges.append(item)
    return {
        "scheme": SCHEME,
        "width": round(graph.width, 3),
        "height": round(graph.height, 3),
        "point_cols": graph.point_cols,
        "point_rows": graph.point_rows,
        "data_cells": graph.data_cells,
        "nodes": [{"id": node.id, "x": round(node.x, 3), "y": round(node.y, 3)} for node in graph.nodes],
        "edges": edges,
    }


def write_json(graph: Graph, path: Path) -> None:
    path.write_text(json.dumps(graph_to_json(graph), indent=2) + "\n", encoding="utf-8")


def write_svg(
    graph: Graph,
    path: Path,
    *,
    edge_color: str = "#2730ff",
    node_color: str = "#f39a18",
    node_stroke_color: str = "#d67900",
    background_color: str = "#ffffff",
) -> None:
    nodes_by_id = {node.id: node for node in graph.nodes}
    line_width = max(0.54, min(graph.width, graph.height) / 760.0)
    nominal_spacing = math.sqrt((graph.width * graph.height) / max(1, len(graph.nodes)))
    node_radius = max(1.35, min(3.35, nominal_spacing * 0.15))
    edge_color = html.escape(edge_color, quote=True)
    node_color = html.escape(node_color, quote=True)
    node_stroke_color = html.escape(node_stroke_color, quote=True)
    background_color = html.escape(background_color, quote=True)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{graph.width:.1f}" '
            f'height="{graph.height:.1f}" viewBox="0 0 {graph.width:.1f} {graph.height:.1f}" '
            f'data-scheme="{SCHEME}">'
        ),
        "  <title>Lattice Bloom encoded graph</title>",
        (
            "  <metadata>"
            + html.escape(json.dumps({"scheme": SCHEME, "plaintext_stored": False}))
            + "</metadata>"
        ),
        f'  <rect width="100%" height="100%" fill="{background_color}"/>',
        f'  <g id="edges" stroke="{edge_color}" stroke-linecap="round">',
    ]
    for edge in graph.edges:
        a = nodes_by_id[edge.u]
        b = nodes_by_id[edge.v]
        opacity = 0.25 + 0.68 * edge.weight
        stroke_width = line_width * (0.58 + 0.72 * edge.weight)
        data_attrs = ""
        if edge.data_cell is not None:
            data_attrs = (
                f' data-cell="{edge.data_cell}" data-slot="{edge.data_slot}"'
                f' data-bit="{edge.data_bit}"'
            )
        parts.append(
            f'    <line x1="{a.x:.3f}" y1="{a.y:.3f}" x2="{b.x:.3f}" y2="{b.y:.3f}" '
            f'stroke-width="{stroke_width:.3f}" stroke-opacity="{opacity:.3f}" '
            f'data-u="{edge.u}" data-v="{edge.v}" data-weight="{edge.weight:.4f}"{data_attrs}/>'
        )
    parts.extend(
        [
            "  </g>",
            f'  <g id="nodes" fill="{node_color}" stroke="{node_stroke_color}" stroke-width="{line_width * 0.44:.3f}">',
        ]
    )
    for node in graph.nodes:
        parts.append(f'    <circle id="{node.id}" cx="{node.x:.3f}" cy="{node.y:.3f}" r="{node_radius:.3f}"/>')
    parts.extend(["  </g>", "</svg>", ""])
    path.write_text("\n".join(parts), encoding="utf-8")


def add_data_score(scores: dict[tuple[int, int, int], float], cell: int, slot: int, bit: int, weight: float) -> None:
    key = (cell, slot, bit)
    scores[key] = max(scores.get(key, 0.0), weight)


def load_graph_json(path: Path) -> dict[tuple[int, int, int], float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("scheme") != SCHEME:
        raise ValueError(f"expected {SCHEME} JSON")
    scores: dict[tuple[int, int, int], float] = {}
    for edge in raw.get("edges", []):
        if "data_cell" not in edge:
            continue
        add_data_score(
            scores,
            int(edge["data_cell"]),
            int(edge["data_slot"]),
            int(edge["data_bit"]),
            float(edge.get("weight", 0.0)),
        )
    return scores


def load_graph_svg(path: Path) -> dict[tuple[int, int, int], float]:
    root = ET.parse(path).getroot()
    scores: dict[tuple[int, int, int], float] = {}
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag != "line" or "data-cell" not in elem.attrib:
            continue
        add_data_score(
            scores,
            int(elem.attrib["data-cell"]),
            int(elem.attrib["data-slot"]),
            int(elem.attrib["data-bit"]),
            float(elem.attrib.get("data-weight", "0")),
        )
    return scores


def read_cell_nibble(cell: int, scores: dict[tuple[int, int, int], float]) -> int:
    nibble = 0
    for slot in range(4):
        zero_score = scores.get((cell, slot, 0), 0.0)
        one_score = scores.get((cell, slot, 1), 0.0)
        if abs(zero_score - one_score) < 1e-9:
            raise ValueError(f"ambiguous graph encoding at cell {cell} slot {slot}")
        bit = 1 if one_score > zero_score else 0
        nibble = (nibble << 1) | bit
    return nibble


def decode_scores(scores: dict[tuple[int, int, int], float]) -> str:
    if not scores:
        raise ValueError("no lattice-bloom data edges found")
    cells = sorted({cell for cell, _slot, _bit in scores})
    if len(cells) < HEADER_CELLS:
        raise ValueError(f"graph needs at least {HEADER_CELLS} encoded cells for a header")

    header_nibbles = [read_cell_nibble(cell, scores) for cell in range(HEADER_CELLS)]
    header = parse_header(nibbles_to_bytes(header_nibbles)[:HEADER_LEN])

    tail_nibble_count = (header.payload_len + CRC_LEN) * 2
    available_data_cells = [cell for cell in cells if cell >= HEADER_CELLS]
    if len(available_data_cells) < tail_nibble_count:
        raise ValueError("graph does not have enough cells for the encoded payload")
    perm_rng = random.Random(header.seed ^ PERMUTE_XOR)
    perm_rng.shuffle(available_data_cells)

    tail_nibbles = [read_cell_nibble(cell, scores) for cell in available_data_cells[:tail_nibble_count]]
    tail = nibbles_to_bytes(tail_nibbles)[: header.payload_len + CRC_LEN]
    return parse_tail(header, tail)


def decode_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        scores = load_graph_json(path)
    elif suffix == ".svg":
        scores = load_graph_svg(path)
    else:
        raise ValueError("input must be a .svg or .json graph file")
    return decode_scores(scores)


def encode_command(args: argparse.Namespace) -> int:
    if args.stdin:
        text = sys.stdin.read()
    elif args.text is not None:
        text = args.text
    else:
        raise SystemExit("encode needs TEXT or --stdin")

    graph = build_graph(
        text,
        cell_size=args.cell_size,
        min_cells=args.min_cells,
        padding=args.padding,
        unit_range=args.unit_range,
        window=args.window,
        norm_radius=args.norm_radius,
        variant_strength=args.variant_strength,
    )
    output = Path(args.output)
    if output.suffix.lower() != ".svg":
        raise SystemExit("SVG output path must end in .svg")
    write_svg(
        graph,
        output,
        edge_color=args.edge_color,
        node_color=args.node_color,
        node_stroke_color=args.node_stroke_color,
        background_color=args.background_color,
    )
    if args.json:
        write_json(graph, Path(args.json))

    print(
        f"encoded {len(text)} characters into {output} "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges, {graph.data_cells} data cells)"
    )
    return 0


def decode_command(args: argparse.Namespace) -> int:
    print(decode_path(Path(args.input)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode/decode text as dense lattice-bloom graph SVGs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("encode", help="encode text into a graph SVG")
    encode.add_argument("text", nargs="?", help="text to encode")
    encode.add_argument("--stdin", action="store_true", help="read text from stdin")
    encode.add_argument("-o", "--output", default="encoded.svg", help="output SVG path")
    encode.add_argument("--json", help="also write a JSON graph file")
    encode.add_argument("--cell-size", type=float, default=52.0, help="unit-distance scale in SVG units")
    encode.add_argument("--min-cells", type=int, default=120, help="minimum encoded cells, including padding")
    encode.add_argument("--padding", type=int, default=24, help="extra random padding cells after the packet")
    encode.add_argument("--unit-range", type=int, default=2, help="coefficient range N for a,b,c,d in {-N,...,N}")
    encode.add_argument(
        "--window",
        choices=WINDOW_CHOICES,
        default="seeded",
        help="coefficient window: seeded text-varying default, norm/bounded-norm for |z| < R, or box for the exact finite box",
    )
    encode.add_argument("--norm-radius", type=float, default=4.0, help="radius R used by --window norm")
    encode.add_argument("--edge-color", default="#2730ff", help="SVG color for graph edges")
    encode.add_argument("--node-color", default="#f39a18", help="SVG fill color for graph vertices")
    encode.add_argument("--node-stroke-color", default="#d67900", help="SVG outline color for graph vertices")
    encode.add_argument("--background-color", default="#ffffff", help="SVG background color")
    encode.add_argument(
        "--variant-strength",
        type=float,
        default=0.75,
        help="seeded-mode visual variation from 0.0 exact box to 1.0 strong polydisc window",
    )
    encode.set_defaults(func=encode_command)

    decode = subparsers.add_parser("decode", help="decode a generated SVG or JSON graph")
    decode.add_argument("input", help="input .svg or .json graph file")
    decode.set_defaults(func=decode_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

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
import heapq
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


SCHEME = "lattice-bloom-v2"
MAGIC = b"LBM2"
VERSION = 2
FLAGS = 0
FLAG_ZLIB = 1
SEED_LEN = 8
HEADER_LEN = 4 + 1 + 1 + SEED_LEN + 4
CRC_LEN = 4
HEADER_CELLS = HEADER_LEN * 2
RING_COUNT = 12
PERMUTE_XOR = 0xC0DEC0DEC0DEC0DE
LAYOUT_XOR = 0x9E3779B97F4A7C15
MESH_XOR = 0xD1B54A32D192ED03

CHORD_PAIRS = [
    ((0, 5), (1, 6)),
    ((2, 7), (3, 8)),
    ((4, 9), (5, 10)),
    ((6, 11), (7, 0)),
]

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


def choose_point_grid(point_count: int, point_cols: int | None) -> tuple[int, int]:
    if point_cols is None:
        point_cols = max(16, math.ceil(math.sqrt(point_count * 0.943)))
    point_rows = math.ceil(point_count / point_cols)
    return point_cols, point_rows


def make_point_field(
    point_count: int,
    *,
    point_cols: int | None,
    spacing: float,
    seed: int,
) -> tuple[list[Node], int, int, set[int]]:
    rng = random.Random(seed ^ LAYOUT_XOR)
    point_cols, point_rows = choose_point_grid(point_count, point_cols)
    margin = spacing * 0.64
    jitter = spacing * 0.030
    y_step = spacing * 1.025
    nodes: list[Node] = []
    for index in range(point_count):
        row, col = divmod(index, point_cols)
        x = margin + col * spacing + (row % 2) * spacing * 0.50
        y = margin + row * y_step
        x += math.sin(row * 0.77 + col * 0.23) * spacing * 0.018
        y += math.cos(col * 0.61 - row * 0.18) * spacing * 0.016
        x += rng.uniform(-jitter, jitter)
        y += rng.uniform(-jitter, jitter)
        nodes.append(Node(f"p{index}", x, y))

    max_x = max(node.x for node in nodes)
    max_y = max(node.y for node in nodes)
    swirl_centers: list[tuple[float, float, float]] = []
    for row in range(1, 5):
        for col in range(1, 5):
            if rng.random() < 0.72:
                swirl_centers.append(
                    (
                        (col / 5.0) * max_x + rng.uniform(-spacing * 0.55, spacing * 0.55),
                        (row / 5.0) * max_y + rng.uniform(-spacing * 0.55, spacing * 0.55),
                        rng.choice([-1.0, 1.0]),
                    )
                )

    warped: list[Node] = []
    for node in nodes:
        shift_x = 0.0
        shift_y = 0.0
        for sx, sy, direction in swirl_centers:
            dx = node.x - sx
            dy = node.y - sy
            distance = math.hypot(dx, dy)
            if distance < 1e-6:
                continue
            influence = math.exp(-((distance / (spacing * 3.15)) ** 2))
            radial = math.sin(distance / spacing * math.tau * 0.42) * spacing * 0.060 * influence
            tangent = math.cos(distance / spacing * math.tau * 0.31) * spacing * 0.045 * influence * direction
            ux = dx / distance
            uy = dy / distance
            shift_x += ux * radial - uy * tangent
            shift_y += uy * radial + ux * tangent
        warped.append(Node(node.id, node.x + shift_x, node.y + shift_y))
    nodes = warped

    min_x = min(node.x for node in nodes)
    min_y = min(node.y for node in nodes)
    translated = [Node(node.id, node.x - min_x + margin, node.y - min_y + margin) for node in nodes]
    motif_centers = {
        index
        for index in range(point_count)
        if 1 < divmod(index, point_cols)[0] < point_rows - 2 and 1 < divmod(index, point_cols)[1] < point_cols - 2
    }
    return translated, point_cols, point_rows, motif_centers


def load_style_template(path: Path) -> tuple[list[Node], set[int], dict[tuple[str, str], float], int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    node_items = raw.get("nodes", {}).get("all") or raw.get("nodes", {}).get("sample") or []
    if not node_items:
        raise ValueError(f"{path} does not contain extracted nodes; rerun analyze_reference.py")

    nodes = [Node(f"p{index}", float(item["x"]), float(item["y"])) for index, item in enumerate(node_items)]
    preferred = {
        int(item["i"])
        for item in raw.get("degree", {}).get("high_degree_sample", [])
        if 0 <= int(item["i"]) < len(nodes)
    }
    if not preferred:
        preferred = set(range(min(len(nodes), 120)))

    visual_edges: dict[tuple[str, str], float] = {}
    for item in raw.get("edges", {}).get("all", []):
        u_index = int(item["u"])
        v_index = int(item["v"])
        if not (0 <= u_index < len(nodes) and 0 <= v_index < len(nodes)):
            continue
        coverage = float(item.get("coverage", 0.45))
        length = float(item.get("length", 40.0))
        length_bias = 1.0 if length < 55.0 else 0.82
        weight = max(0.18, min(0.72, (0.18 + coverage * 0.62) * length_bias))
        add_visual_edge(visual_edges, nodes[u_index].id, nodes[v_index].id, weight)

    image = raw.get("image", {})
    width = int(image.get("width", 0) or 0)
    height = int(image.get("height", 0) or 0)
    return nodes, preferred, visual_edges, width, height


def unit_difference_vectors() -> list[tuple[int, int, int, int]]:
    vectors: list[tuple[int, int, int, int]] = []
    for da, db, dc, dd in itertools.product(range(-1, 2), repeat=4):
        if da == db == dc == dd == 0:
            continue
        # For z = a + bi + c rho + d i rho and rho = (-1 + sqrt(3)i) / 2,
        # write 2 Re(z) = R0 + R1 sqrt(3), 2 Im(z) = I0 + I1 sqrt(3).
        # Unit length is then the exact integer system below.
        r0 = 2 * da - dc
        r1 = -dd
        i0 = 2 * db - dd
        i1 = dc
        if r0 * r1 + i0 * i1 != 0:
            continue
        if r0 * r0 + 3 * r1 * r1 + i0 * i0 + 3 * i1 * i1 == 4:
            vectors.append((da, db, dc, dd))
    return vectors


def project_coeffs(coeffs: tuple[int, ...], basis: list[complex]) -> complex:
    return sum(value * direction for value, direction in zip(coeffs, basis))


def stable_unit(seed: int, *parts: object) -> float:
    digest = hashlib.sha256(f"{seed}:{parts!r}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def angular_distance_mod_pi(a: float, b: float) -> float:
    diff = abs((a - b + math.pi / 2.0) % math.pi - math.pi / 2.0)
    return diff


def make_projection_basis(seed: int, variant_strength: float) -> list[complex]:
    strength = max(0.0, min(1.0, variant_strength))
    if strength <= 1e-9:
        rho = complex(-0.5, math.sqrt(3.0) / 2.0)
        return [1.0 + 0.0j, 1j, rho, 1j * rho]

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
    variant_strength: float,
) -> tuple[list[Node], set[int], dict[tuple[str, str], float], dict[int, list[int]]]:
    strength = max(0.0, min(1.0, variant_strength))
    basis = make_projection_basis(seed, strength)
    raw_points: list[tuple[float, float, tuple[int, ...]]] = []
    seen: set[tuple[float, float]] = set()
    for coeffs in choose_coefficients(range_limit, seed, strength, basis):
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
            add_visual_edge(visual_edges, nodes[i].id, nodes[j].id, 0.54 + edge_noise * 0.22 * strength)
            incident[i].append(j)
            incident[j].append(i)

    for center_index, neighbors in incident.items():
        center = nodes[center_index]
        neighbors.sort(key=lambda index: math.atan2(nodes[index].y - center.y, nodes[index].x - center.x))

    preferred = {index for index, neighbors in incident.items() if len(neighbors) >= 8}
    return nodes, preferred, visual_edges, incident


def circularize_neighborhoods(
    nodes: list[Node],
    center_indices: list[int],
    spacing: float,
    seed: int,
) -> tuple[list[Node], set[int]]:
    rng = random.Random(seed ^ 0xA5D7C931E5B85A21)
    result = list(nodes)
    protected = set(center_indices)
    moved: set[int] = set()
    motif_centers: list[int] = []
    candidates = list(center_indices)
    rng.shuffle(candidates)
    target_motifs = min(30, max(20, len(nodes) // 17))
    min_center_distance = spacing * 2.35

    for center_index in candidates:
        center = result[center_index]
        if any(math.hypot(center.x - result[other].x, center.y - result[other].y) < min_center_distance for other in motif_centers):
            continue
        motif_centers.append(center_index)
        if len(motif_centers) >= target_motifs:
            break

    for motif_number, center_index in enumerate(motif_centers):
        center = result[center_index]
        nearby = [
            (math.hypot(center.x - node.x, center.y - node.y), index)
            for index, node in enumerate(result)
            if index != center_index and index not in protected and index not in moved
        ]
        nearby.sort(key=lambda item: item[0])
        ring_size = rng.randint(5, 6)
        chosen = [index for _distance, index in nearby[: ring_size * 3]]
        rng.shuffle(chosen)
        chosen = chosen[:ring_size]
        slots = [0, 2, 4, 6, 8, 10]
        rng.shuffle(slots)
        slots = sorted(slots[:ring_size])
        radius = spacing * rng.uniform(0.52, 0.58)
        phase = rng.uniform(0, math.tau)
        for index, slot in zip(chosen, slots):
            angle = phase + slot * math.tau / 12.0
            target_x = center.x + math.cos(angle) * radius + rng.uniform(-spacing * 0.025, spacing * 0.025)
            target_y = center.y + math.sin(angle) * radius + rng.uniform(-spacing * 0.025, spacing * 0.025)
            blend = rng.uniform(0.86, 1.0)
            original = result[index]
            result[index] = Node(
                result[index].id,
                original.x * (1.0 - blend) + target_x * blend,
                original.y * (1.0 - blend) + target_y * blend,
            )
            moved.add(index)

    return result, set(motif_centers)


def point_index(node: Node) -> int:
    return int(node.id[1:])


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


def choose_ring_indices(nodes: list[Node], center_index: int, spacing: float, seed: int, cell: int) -> list[int]:
    center = nodes[center_index]
    phase_rng = random.Random(seed ^ (cell * 0x9E3779B1))
    phase = phase_rng.uniform(0, math.tau)
    radius = spacing * phase_rng.uniform(1.15, 1.72)
    chosen: list[int] = []

    for ring in range(RING_COUNT):
        angle = phase + ring * math.tau / RING_COUNT
        target_x = center.x + math.cos(angle) * radius
        target_y = center.y + math.sin(angle) * radius
        candidates = sorted(
            (
                ((node.x - target_x) ** 2 + (node.y - target_y) ** 2, index)
                for index, node in enumerate(nodes)
                if index != center_index and index not in chosen
            ),
            key=lambda item: item[0],
        )
        chosen.append(candidates[0][1])

    return chosen


def nearest_indices(nodes: list[Node], source_index: int, count: int) -> list[int]:
    source = nodes[source_index]
    nearest = heapq.nsmallest(
        count,
        (
            ((source.x - target.x) ** 2 + (source.y - target.y) ** 2, target_index)
            for target_index, target in enumerate(nodes)
            if target_index != source_index
        ),
    )
    return [index for _distance, index in nearest]


def build_graph(
    text: str,
    *,
    cols: int | None = None,
    cell_size: float = 52.0,
    min_cells: int = 120,
    padding: int = 24,
    style_json: Path | None = None,
    unit_range: int = 2,
    variant_strength: float = 0.75,
) -> Graph:
    header, tail, seed, _normalized = make_parts(text)
    pad_rng = random.Random(seed ^ 0x517CC1B727220A95)
    mesh_rng = random.Random(seed ^ MESH_XOR)

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

    style_visual_edges: dict[tuple[str, str], float] = {}
    unit_incident: dict[int, list[int]] = {}
    if style_json is not None:
        nodes, round_motif_centers, style_visual_edges, _style_width, _style_height = load_style_template(style_json)
        point_cols = 0
        point_rows = 0
    else:
        effective_unit_range = unit_range
        while True:
            nodes, round_motif_centers, style_visual_edges, unit_incident = make_unit_distance_substrate(
                effective_unit_range,
                cell_size,
                seed,
                variant_strength,
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

    if style_json is not None:
        synthetic_cover_scale = 0.40
        for source_index in range(len(nodes)):
            near = nearest_indices(nodes, source_index, 10)
            for rank, target_index in enumerate(near[:7]):
                chance = 0.82 - rank * 0.055
                if mesh_rng.random() < chance:
                    add_visual_edge(
                        visual_edges,
                        nodes[source_index].id,
                        nodes[target_index].id,
                        mesh_rng.uniform(0.20, 0.44) * synthetic_cover_scale,
                    )

    for cell, center_index in enumerate(center_indices):
        center_id = nodes[center_index].id
        if unit_incident:
            neighbor_indices = unit_incident[center_index]
            if len(neighbor_indices) < 8:
                neighbor_indices = nearest_indices(nodes, center_index, 8)
            nibble = nibbles[cell]
            for slot in range(4):
                bit = (nibble >> (3 - slot)) & 1
                pair = (neighbor_indices[(2 * slot) % len(neighbor_indices)], neighbor_indices[(2 * slot + 1) % len(neighbor_indices)])
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
            continue

        ring_indices = choose_ring_indices(nodes, center_index, cell_size, seed, cell)
        ring_ids = [nodes[index].id for index in ring_indices]
        # The visual bloom is made from shared lattice points, not separate
        # satellite nodes. That keeps the orange dots evenly distributed.
        is_round_motif = center_index in round_motif_centers
        motif_indices = nearest_indices(nodes, center_index, 12 if is_round_motif else 8)
        motif_indices.sort(key=lambda index: math.atan2(nodes[index].y - nodes[center_index].y, nodes[index].x - nodes[center_index].x))
        for motif_index in motif_indices:
            if mesh_rng.random() < (0.90 if is_round_motif else 0.56):
                add_visual_edge(
                    visual_edges,
                    center_id,
                    nodes[motif_index].id,
                    mesh_rng.uniform(0.50, 0.76) if is_round_motif else mesh_rng.uniform(0.24, 0.44),
                )
        for ring in range(len(motif_indices)):
            if mesh_rng.random() < (0.58 if is_round_motif else 0.28):
                add_visual_edge(
                    visual_edges,
                    nodes[motif_indices[ring]].id,
                    nodes[motif_indices[(ring + 1) % len(motif_indices)]].id,
                    mesh_rng.uniform(0.28, 0.46) if is_round_motif else mesh_rng.uniform(0.16, 0.32),
                )
            if mesh_rng.random() < (0.34 if is_round_motif else 0.12):
                add_visual_edge(
                    visual_edges,
                    nodes[motif_indices[ring]].id,
                    nodes[motif_indices[(ring + 2) % len(motif_indices)]].id,
                    mesh_rng.uniform(0.22, 0.40) if is_round_motif else mesh_rng.uniform(0.14, 0.26),
                )

        for ring_id in ring_ids:
            if mesh_rng.random() < (0.46 if is_round_motif else 0.20):
                add_visual_edge(visual_edges, center_id, ring_id, mesh_rng.uniform(0.28, 0.48))
        for ring in range(RING_COUNT):
            if mesh_rng.random() < 0.28:
                add_visual_edge(
                    visual_edges,
                    ring_ids[ring],
                    ring_ids[(ring + 1) % RING_COUNT],
                    mesh_rng.uniform(0.18, 0.36),
                )
            if mesh_rng.random() < 0.38:
                add_visual_edge(
                    visual_edges,
                    ring_ids[ring],
                    ring_ids[(ring + 2) % RING_COUNT],
                    mesh_rng.uniform(0.17, 0.34),
                )

        nibble = nibbles[cell]
        for slot, pair in enumerate(CHORD_PAIRS):
            bit = (nibble >> (3 - slot)) & 1
            for candidate_bit, (a, b) in enumerate(pair):
                weight = DATA_STRONG_WEIGHT if candidate_bit == bit else DATA_WEAK_WEIGHT
                data_edges.append(
                    Edge(
                        ring_ids[a],
                        ring_ids[b],
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


def write_svg(graph: Graph, path: Path) -> None:
    nodes_by_id = {node.id: node for node in graph.nodes}
    line_width = max(0.54, min(graph.width, graph.height) / 760.0)
    nominal_spacing = math.sqrt((graph.width * graph.height) / max(1, len(graph.nodes)))
    node_radius = max(1.35, min(3.35, nominal_spacing * 0.15))

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
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
        '  <g id="edges" stroke="#2730ff" stroke-linecap="round">',
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
            f'  <g id="nodes" fill="#f39a18" stroke="#d67900" stroke-width="{line_width * 0.44:.3f}">',
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

    style_json = Path(args.style_json) if args.style_json else None

    graph = build_graph(
        text,
        cols=args.cols,
        cell_size=args.cell_size,
        min_cells=args.min_cells,
        padding=args.padding,
        style_json=style_json,
        unit_range=args.unit_range,
        variant_strength=args.variant_strength,
    )
    output = Path(args.output)
    if output.suffix.lower() != ".svg":
        raise SystemExit("SVG output path must end in .svg")
    write_svg(graph, output)
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
    encode.add_argument("--cols", type=int, help="number of point-field columns")
    encode.add_argument("--cell-size", type=float, default=52.0, help="unit-distance scale in SVG units")
    encode.add_argument("--min-cells", type=int, default=120, help="minimum encoded cells, including padding")
    encode.add_argument("--padding", type=int, default=24, help="extra random padding cells after the packet")
    encode.add_argument("--style-json", help="reference analysis JSON to use as the visual substrate")
    encode.add_argument("--unit-range", type=int, default=2, help="coefficient range N for a,b,c,d in {-N,...,N}")
    encode.add_argument(
        "--variant-strength",
        type=float,
        default=0.75,
        help="seeded visual variation from 0.0 exact box to 1.0 strong polydisc window",
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

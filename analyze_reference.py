#!/usr/bin/env python3
"""Extract rough graph statistics from the reference PNG without external deps."""

from __future__ import annotations

import json
import math
import statistics
import struct
import sys
import zlib
from collections import deque
from pathlib import Path


PNG_SIG = b"\x89PNG\r\n\x1a\n"


def read_png_rgb(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    raw = path.read_bytes()
    if not raw.startswith(PNG_SIG):
        raise ValueError("not a PNG")
    pos = len(PNG_SIG)
    width = height = bit_depth = color_type = None
    data = bytearray()
    interlace = None
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        pos += 4
        chunk_type = raw[pos : pos + 4]
        pos += 4
        chunk = raw[pos : pos + length]
        pos += length + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filter, interlace = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            data.extend(chunk)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("missing IHDR")
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(f"unsupported PNG format bit_depth={bit_depth} color_type={color_type} interlace={interlace}")

    channels = 3 if color_type == 2 else 4
    stride = width * channels
    inflated = zlib.decompress(bytes(data))
    pixels: list[tuple[int, int, int]] = []
    prev = [0] * stride
    idx = 0
    for _row in range(height):
        filter_type = inflated[idx]
        idx += 1
        scan = list(inflated[idx : idx + stride])
        idx += stride
        recon = [0] * stride
        for i, value in enumerate(scan):
            left = recon[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filter_type == 0:
                out = value
            elif filter_type == 1:
                out = value + left
            elif filter_type == 2:
                out = value + up
            elif filter_type == 3:
                out = value + ((left + up) // 2)
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                pr = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                out = value + pr
            else:
                raise ValueError(f"bad filter {filter_type}")
            recon[i] = out & 0xFF
        prev = recon
        for x in range(width):
            base = x * channels
            pixels.append((recon[base], recon[base + 1], recon[base + 2]))
    return width, height, pixels


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def histogram(values: list[float], bins: list[float]) -> list[int]:
    counts = [0] * (len(bins) - 1)
    for value in values:
        for index in range(len(bins) - 1):
            if bins[index] <= value < bins[index + 1]:
                counts[index] += 1
                break
    return counts


def quadrat_stats(nodes: list[dict], width: int, height: int, cols: int = 8, rows: int = 8) -> dict:
    counts = [0] * (cols * rows)
    for node in nodes:
        col = min(cols - 1, max(0, int(node["x"] / width * cols)))
        row = min(rows - 1, max(0, int(node["y"] / height * rows)))
        counts[row * cols + col] += 1
    mean = statistics.mean(counts) if counts else 0.0
    stdev = statistics.pstdev(counts) if counts else 0.0
    return {
        "grid": [counts[row * cols : (row + 1) * cols] for row in range(rows)],
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean if mean else 0.0,
        "min": min(counts, default=0),
        "max": max(counts, default=0),
    }


def nearest_k_distances(nodes: list[dict], k: int) -> dict:
    by_k: list[list[float]] = [[] for _ in range(k)]
    all_pair_short: list[float] = []
    for i, node in enumerate(nodes):
        distances = sorted(
            math.hypot(node["x"] - other["x"], node["y"] - other["y"])
            for j, other in enumerate(nodes)
            if i != j
        )
        for index in range(min(k, len(distances))):
            by_k[index].append(distances[index])
        all_pair_short.extend(distance for distance in distances[: min(8, len(distances))])
    return {
        "k_medians": [percentile(values, 50) for values in by_k],
        "k_p10": [percentile(values, 10) for values in by_k],
        "k_p90": [percentile(values, 90) for values in by_k],
        "short_distance_bins": {
            "bins": [0, 8, 12, 16, 20, 24, 28, 32, 40, 56, 80],
            "counts": histogram(all_pair_short, [0, 8, 12, 16, 20, 24, 28, 32, 40, 56, 80]),
        },
    }


def row_column_spacing(nodes: list[dict], width: int, height: int) -> dict:
    y_sorted = sorted(node["y"] for node in nodes)
    x_sorted = sorted(node["x"] for node in nodes)
    y_gaps = [b - a for a, b in zip(y_sorted, y_sorted[1:]) if b - a > 2.0]
    x_gaps = [b - a for a, b in zip(x_sorted, x_sorted[1:]) if b - a > 2.0]

    row_buckets: dict[int, int] = {}
    col_buckets: dict[int, int] = {}
    for node in nodes:
        row_buckets[round(node["y"] / 4) * 4] = row_buckets.get(round(node["y"] / 4) * 4, 0) + 1
        col_buckets[round(node["x"] / 4) * 4] = col_buckets.get(round(node["x"] / 4) * 4, 0) + 1

    return {
        "x_gap_median": percentile(x_gaps, 50),
        "y_gap_median": percentile(y_gaps, 50),
        "strong_rows": sorted(row_buckets.items(), key=lambda item: item[1], reverse=True)[:16],
        "strong_cols": sorted(col_buckets.items(), key=lambda item: item[1], reverse=True)[:16],
    }


def component_centers(mask: list[bool], width: int, height: int, min_area: int, max_area: int) -> list[dict]:
    seen = bytearray(width * height)
    centers: list[dict] = []
    for start, is_on in enumerate(mask):
        if not is_on or seen[start]:
            continue
        q = deque([start])
        seen[start] = 1
        count = 0
        sx = sy = 0.0
        min_x = width
        min_y = height
        max_x = max_y = 0
        while q:
            idx = q.popleft()
            x = idx % width
            y = idx // width
            count += 1
            sx += x
            sy += y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                ni = ny * width + nx
                if mask[ni] and not seen[ni]:
                    seen[ni] = 1
                    q.append(ni)
        if min_area <= count <= max_area:
            centers.append(
                {
                    "x": sx / count,
                    "y": sy / count,
                    "area": count,
                    "bbox": [min_x, min_y, max_x, max_y],
                }
            )
    return centers


def sample_line(mask: list[bool], width: int, p: dict, q: dict) -> tuple[float, int]:
    dx = q["x"] - p["x"]
    dy = q["y"] - p["y"]
    distance = math.hypot(dx, dy)
    steps = max(6, min(80, int(distance * 0.9)))
    hits = 0
    total = 0
    for step in range(1, steps):
        t = step / steps
        x = int(round(p["x"] + dx * t))
        y = int(round(p["y"] + dy * t))
        total += 1
        found = False
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                if 0 <= xx < width and 0 <= yy and yy * width + xx < len(mask) and mask[yy * width + xx]:
                    found = True
                    break
            if found:
                break
        if found:
            hits += 1
    return (hits / total if total else 0.0), total


def analyze(path: Path, out_path: Path) -> dict:
    width, height, pixels = read_png_rgb(path)

    orange_mask = []
    blue_mask = []
    for r, g, b in pixels:
        orange = r > 175 and 80 < g < 180 and b < 105 and r > g + 18 and g > b + 30
        blue = b > 130 and b > r + 34 and b > g + 28 and not orange
        orange_mask.append(orange)
        blue_mask.append(blue)

    nodes = component_centers(orange_mask, width, height, min_area=5, max_area=140)
    nodes.sort(key=lambda item: (item["y"], item["x"]))

    nn_distances = []
    for i, node in enumerate(nodes):
        best = min(
            (
                math.hypot(node["x"] - other["x"], node["y"] - other["y"])
                for j, other in enumerate(nodes)
                if i != j
            ),
            default=0.0,
        )
        if best:
            nn_distances.append(best)

    pair_edges = []
    degrees = [0] * len(nodes)
    for i, node in enumerate(nodes):
        candidates = []
        for j, other in enumerate(nodes):
            if i >= j:
                continue
            d = math.hypot(node["x"] - other["x"], node["y"] - other["y"])
            if 5.0 <= d <= 74.0:
                candidates.append((d, j, other))
        candidates.sort(key=lambda item: item[0])
        for d, j, other in candidates[:42]:
            coverage, samples = sample_line(blue_mask, width, node, other)
            threshold = 0.39 if d < 32 else 0.46
            if coverage >= threshold:
                angle = (math.degrees(math.atan2(other["y"] - node["y"], other["x"] - node["x"])) + 180) % 180
                pair_edges.append({"u": i, "v": j, "length": d, "coverage": coverage, "angle": angle})
                degrees[i] += 1
                degrees[j] += 1

    edge_lengths = [edge["length"] for edge in pair_edges]
    angles = [edge["angle"] for edge in pair_edges]
    angle_bins = [0] * 12
    for angle in angles:
        angle_bins[min(11, int(angle / 15))] += 1

    high_degree = [
        {"i": i, "x": nodes[i]["x"], "y": nodes[i]["y"], "degree": degree}
        for i, degree in enumerate(degrees)
        if degree >= percentile(degrees, 90)
    ]

    result = {
        "image": {"path": str(path), "width": width, "height": height},
        "mask_pixels": {
            "orange": sum(orange_mask),
            "blue": sum(blue_mask),
            "orange_fraction": sum(orange_mask) / len(orange_mask),
            "blue_fraction": sum(blue_mask) / len(blue_mask),
        },
        "nodes": {
            "count": len(nodes),
            "all": [
                {
                    "x": round(node["x"], 4),
                    "y": round(node["y"], 4),
                    "area": node["area"],
                    "bbox": node["bbox"],
                }
                for node in nodes
            ],
            "sample": nodes[:20],
            "area": {
                "median": percentile([node["area"] for node in nodes], 50),
                "p10": percentile([node["area"] for node in nodes], 10),
                "p90": percentile([node["area"] for node in nodes], 90),
            },
            "nearest_distance": {
                "median": percentile(nn_distances, 50),
                "p10": percentile(nn_distances, 10),
                "p90": percentile(nn_distances, 90),
            },
            "nearest_k": nearest_k_distances(nodes, 6),
            "quadrats_8x8": quadrat_stats(nodes, width, height, 8, 8),
            "row_column_spacing": row_column_spacing(nodes, width, height),
        },
        "edges": {
            "estimated_count": len(pair_edges),
            "all": [
                {
                    "u": edge["u"],
                    "v": edge["v"],
                    "length": round(edge["length"], 4),
                    "coverage": round(edge["coverage"], 4),
                    "angle": round(edge["angle"], 4),
                }
                for edge in pair_edges
            ],
            "length": {
                "median": percentile(edge_lengths, 50),
                "p10": percentile(edge_lengths, 10),
                "p90": percentile(edge_lengths, 90),
                "max": max(edge_lengths, default=0),
            },
            "angle_bins_15deg": angle_bins,
            "sample": pair_edges[:40],
        },
        "degree": {
            "median": percentile(degrees, 50),
            "p10": percentile(degrees, 10),
            "p90": percentile(degrees, 90),
            "max": max(degrees, default=0),
            "high_degree_sample": high_degree[:30],
        },
        "summary": {},
    }
    result["summary"] = {
        "node_density_per_10000_px": len(nodes) / (width * height) * 10000,
        "edge_to_node_ratio": len(pair_edges) / max(1, len(nodes)),
        "blue_to_orange_pixel_ratio": sum(blue_mask) / max(1, sum(orange_mask)),
        "median_spacing_to_edge_length": percentile(edge_lengths, 50) / max(1.0, percentile(nn_distances, 50)),
    }

    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: analyze_reference.py INPUT.png OUTPUT.json")
    result = analyze(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
    print(json.dumps({k: result[k] for k in ("image", "mask_pixels", "summary")}, indent=2))
    print(json.dumps({"nodes": result["nodes"], "edges": result["edges"], "degree": result["degree"]}, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

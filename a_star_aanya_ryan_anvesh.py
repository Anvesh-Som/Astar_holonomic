#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAP_WIDTH = 600
MAP_HEIGHT = 250
XY_RESOLUTION = 0.5
THETA_RESOLUTION_DEG = 30
THETA_BINS = 360 // THETA_RESOLUTION_DEG
GOAL_THRESHOLD = 2.5
DEFAULT_ROBOT_RADIUS = 5.0
DEFAULT_CLEARANCE = 5.0
DEFAULT_STEP_SIZE = 5.0
DEFAULT_TEXT = "TEAM 1"

BASE_MAP_WIDTH = 180.0
BASE_MAP_HEIGHT = 50.0
UNIFORM_SCALE = MAP_WIDTH / BASE_MAP_WIDTH
VERTICAL_OFFSET = (MAP_HEIGHT - BASE_MAP_HEIGHT * UNIFORM_SCALE) / 2.0
HORIZONTAL_OFFSET = 0.0

ACTION_DELTAS_DEG: Tuple[int, ...] = (0, 30, 60, -30, -60)

Rect = Tuple[float, float, float, float]
Polygon = Sequence[Tuple[float, float]]
EllipseRing = Tuple[float, float, float, float, float, float]

FONT_5X7: Dict[str, Sequence[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10011", "10101", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
}
ROUND_GLYPHS = {"0", "6", "8", "9", "O"}


@dataclass(frozen=True)
class State:
    x: float
    y: float
    theta_deg: int


def wrap_theta_deg(theta_deg: float) -> int:
    return int(round(theta_deg)) % 360


def normalize_input_theta(theta_deg: int) -> int:
    if theta_deg % THETA_RESOLUTION_DEG != 0:
        raise ValueError(f"Theta must be a multiple of {THETA_RESOLUTION_DEG} degrees.")
    return theta_deg % 360


def make_state(x: float, y: float, theta_deg: int) -> State:
    return State(float(x), float(y), normalize_input_theta(int(theta_deg)))


def point_in_rect(x: float, y: float, rect: Rect, clearance: float = 0.0) -> bool:
    x0, y0, x1, y1 = rect
    return (x0 - clearance) <= x <= (x1 + clearance) and (y0 - clearance) <= y <= (y1 + clearance)


def point_in_ellipse(x: float, y: float, cx: float, cy: float, rx: float, ry: float, clearance: float = 0.0) -> bool:
    rx_eff = max(rx + clearance, 1e-6)
    ry_eff = max(ry + clearance, 1e-6)
    return ((x - cx) / rx_eff) ** 2 + ((y - cy) / ry_eff) ** 2 <= 1.0


def point_in_ring(
    x: float,
    y: float,
    cx: float,
    cy: float,
    outer_rx: float,
    outer_ry: float,
    inner_rx: float,
    inner_ry: float,
    clearance: float = 0.0,
) -> bool:
    return point_in_ellipse(x, y, cx, cy, outer_rx, outer_ry, clearance) and not point_in_ellipse(
        x, y, cx, cy, inner_rx, inner_ry, -clearance
    )


def point_in_polygon(x: float, y: float, polygon: Polygon, clearance: float = 0.0) -> bool:
    sign = None
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        ex = x2 - x1
        ey = y2 - y1
        nx = -ey
        ny = ex
        norm = math.hypot(nx, ny)
        if norm == 0.0:
            continue
        value = ((x - x1) * nx + (y - y1) * ny) / norm
        if sign is None:
            sign = 1 if value >= 0.0 else -1
        if sign > 0 and value < -clearance:
            return False
        if sign < 0 and value > clearance:
            return False
    return True


def scale_point(x: float, y: float) -> Tuple[float, float]:
    return (HORIZONTAL_OFFSET + x * UNIFORM_SCALE, VERTICAL_OFFSET + y * UNIFORM_SCALE)


def scale_rect(rect: Rect) -> Rect:
    x0, y0 = scale_point(rect[0], rect[1])
    x1, y1 = scale_point(rect[2], rect[3])
    return (x0, y0, x1, y1)


def scale_polygon(polygon: Polygon) -> Polygon:
    return [scale_point(x, y) for x, y in polygon]


def scale_ring(ring: EllipseRing) -> EllipseRing:
    cx, cy = scale_point(ring[0], ring[1])
    return (
        cx,
        cy,
        ring[2] * UNIFORM_SCALE,
        ring[3] * UNIFORM_SCALE,
        ring[4] * UNIFORM_SCALE,
        ring[5] * UNIFORM_SCALE,
    )


class ObstacleMap:
    def __init__(self, text: str, total_clearance: float):
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.total_clearance = total_clearance
        self.text = "".join(ch for ch in text.upper() if ch == " " or ch.isalnum()) or DEFAULT_TEXT

        base_rects, base_rings, base_polygons = self._build_project02_geometry(self.text)
        self.rects = [scale_rect(rect) for rect in base_rects]
        self.rings = [scale_ring(ring) for ring in base_rings]
        self.polygons = [scale_polygon(poly) for poly in base_polygons]

        self.hard_mask = self._rasterize(clearance=0.0)
        self.bloated_mask = self._rasterize(clearance=self.total_clearance)

    def _build_project02_geometry(self, text: str) -> Tuple[List[Rect], List[EllipseRing], List[Polygon]]:
        rects: List[Rect] = []
        rings: List[EllipseRing] = []
        polygons: List[Polygon] = []

        letter_gap = 12.0
        word_gap = 15.0
        left_margin = 20.0
        bottom_margin = 8.0
        glyph_height = 28.0
        cell_h = glyph_height / 7.0
        cell_w = 2.8

        x_cursor = left_margin
        y0 = bottom_margin
        for ch in text:
            if ch == " ":
                x_cursor += word_gap
                continue

            if ch in ROUND_GLYPHS:
                self._append_round_glyph(ch, x_cursor, y0, 14.0, glyph_height, rects, rings)
                x_cursor += 14.0 + letter_gap
                continue

            pattern = FONT_5X7.get(ch, FONT_5X7["E"])
            for row, row_bits in enumerate(pattern):
                for col, bit in enumerate(row_bits):
                    if bit != "1":
                        continue
                    rx0 = x_cursor + col * cell_w
                    ry1 = y0 + glyph_height - row * cell_h
                    ry0 = ry1 - cell_h
                    rects.append((rx0, ry0, rx0 + cell_w, ry1))
            x_cursor += 5.0 * cell_w + letter_gap

        return rects, rings, polygons

    @staticmethod
    def _append_round_glyph(
        ch: str,
        x0: float,
        y0: float,
        width: float,
        height: float,
        rects: List[Rect],
        rings: List[EllipseRing],
    ) -> None:
        cx = x0 + width / 2.0
        cy = y0 + height / 2.0
        outer_rx = width / 2.0
        outer_ry = height / 2.0
        inner_rx = outer_rx * 0.45
        inner_ry = outer_ry * 0.45

        if ch in {"0", "O"}:
            rings.append((cx, cy, outer_rx, outer_ry, inner_rx, inner_ry))
            return

        if ch == "8":
            rings.append((cx, y0 + height * 0.72, outer_rx * 0.78, outer_ry * 0.40, inner_rx * 0.75, inner_ry * 0.42))
            rings.append((cx, y0 + height * 0.28, outer_rx * 0.95, outer_ry * 0.48, inner_rx * 0.90, inner_ry * 0.45))
            return

        if ch == "6":
            rings.append((cx, y0 + height * 0.35, outer_rx, outer_ry * 0.58, inner_rx, inner_ry * 0.85))
            rects.append((x0 + width * 0.08, y0 + height * 0.35, x0 + width * 0.38, y0 + height * 0.96))
            rects.append((x0 + width * 0.18, y0 + height * 0.78, x0 + width * 0.84, y0 + height * 0.96))
            return

        if ch == "9":
            rings.append((cx, y0 + height * 0.65, outer_rx, outer_ry * 0.58, inner_rx, inner_ry * 0.85))
            rects.append((x0 + width * 0.62, y0 + height * 0.04, x0 + width * 0.92, y0 + height * 0.65))
            rects.append((x0 + width * 0.16, y0 + height * 0.04, x0 + width * 0.82, y0 + height * 0.22))
            return

        pattern = FONT_5X7.get(ch, FONT_5X7["0"])
        cell_h = height / 7.0
        cell_w = width / 5.0
        for row, row_bits in enumerate(pattern):
            for col, bit in enumerate(row_bits):
                if bit != "1":
                    continue
                rx0 = x0 + col * cell_w
                ry1 = y0 + height - row * cell_h
                ry0 = ry1 - cell_h
                rects.append((rx0, ry0, rx0 + cell_w, ry1))

    def _point_in_obstacle_geometry(self, x: float, y: float, clearance: float) -> bool:
        for rect in self.rects:
            if point_in_rect(x, y, rect, clearance):
                return True
        for polygon in self.polygons:
            if point_in_polygon(x, y, polygon, clearance):
                return True
        for ring in self.rings:
            if point_in_ring(x, y, *ring, clearance):
                return True
        return False

    def _point_hits_wall(self, x: float, y: float, clearance: float) -> bool:
        return (
            x <= clearance
            or x >= (self.width - 1.0 - clearance)
            or y <= clearance
            or y >= (self.height - 1.0 - clearance)
        )

    def _rasterize(self, clearance: float) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                if self._point_hits_wall(float(x), float(y), clearance) or self._point_in_obstacle_geometry(
                    float(x), float(y), clearance
                ):
                    mask[y, x] = 1
        return mask

    def is_in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.width and 0.0 <= y < self.height

    def is_free_point(self, x: float, y: float) -> bool:
        if not self.is_in_bounds(x, y):
            return False
        ix = min(self.width - 1, max(0, int(round(x))))
        iy = min(self.height - 1, max(0, int(round(y))))
        return not bool(self.bloated_mask[iy, ix])

    def is_motion_valid(self, start: State, end: State) -> bool:
        return self.is_free_point(start.x, start.y) and self.is_free_point(end.x, end.y)


def quantize_xy(value: float, upper_bound: int) -> int:
    index = int(round(value / XY_RESOLUTION))
    return max(0, min(int(upper_bound / XY_RESOLUTION) - 1, index))


def state_to_index(state: State) -> Tuple[int, int, int]:
    iy = quantize_xy(state.y, MAP_HEIGHT)
    ix = quantize_xy(state.x, MAP_WIDTH)
    it = (state.theta_deg // THETA_RESOLUTION_DEG) % THETA_BINS
    return (iy, ix, it)


def euclidean_distance_xy(a: State, b: State) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def state_reaches_target(current: State, target: State) -> bool:
    return euclidean_distance_xy(current, target) <= GOAL_THRESHOLD


def generate_predecessors(current: State, step_size: float) -> Iterable[State]:
    theta_rad = math.radians(current.theta_deg)
    prev_x = current.x - step_size * math.cos(theta_rad)
    prev_y = current.y - step_size * math.sin(theta_rad)

    for delta_deg in ACTION_DELTAS_DEG:
        prev_theta = wrap_theta_deg(current.theta_deg - delta_deg)
        yield State(prev_x, prev_y, prev_theta)


def validate_state(state: State, obstacle_map: ObstacleMap, name: str) -> None:
    if not obstacle_map.is_in_bounds(state.x, state.y):
        raise ValueError(f"{name} must lie inside the map bounds.")
    if not obstacle_map.is_free_point(state.x, state.y):
        raise ValueError(f"{name} lies in obstacle space or clearance space.")


def ask_state(name: str, obstacle_map: ObstacleMap, preset: Optional[Tuple[float, float, int]]) -> State:
    if preset is not None:
        state = make_state(*preset)
        validate_state(state, obstacle_map, name)
        return state

    while True:
        raw = input(f"Enter {name} as x y theta_deg in map frame: ").strip()
        parts = [token for token in raw.replace(",", " ").split() if token]
        if len(parts) != 3:
            print("Invalid format. Example: 40 30 0")
            continue
        try:
            state = make_state(float(parts[0]), float(parts[1]), int(parts[2]))
            validate_state(state, obstacle_map, name)
            return state
        except ValueError as exc:
            print(exc)


def ask_positive_value(prompt: str, preset: Optional[float], low: float, high: float) -> float:
    if preset is not None:
        value = float(preset)
        if not (low <= value <= high):
            raise ValueError(f"{prompt} must be within [{low}, {high}].")
        return value

    while True:
        raw = input(f"{prompt} [{low} to {high}]: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if low <= value <= high:
            return value
        print(f"Value must be within [{low}, {high}].")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backward A* motion model setup for ENPM661 Project 03 - Phase 1")
    parser.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "THETA"), help="start state")
    parser.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "THETA"), help="goal state")
    parser.add_argument("--robot-radius", type=float, default=DEFAULT_ROBOT_RADIUS, help="robot radius in map units")
    parser.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE, help="desired obstacle clearance in map units")
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE, help="step size, 1 <= L <= 10")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Text obstacle string")
    return parser


def run() -> int:
    args = build_argument_parser().parse_args()

    try:
        step_size = ask_positive_value("Step size", args.step_size, 1.0, 10.0)
        robot_radius = ask_positive_value("Robot radius", args.robot_radius, 0.0, 50.0)
        clearance = ask_positive_value("Clearance", args.clearance, 0.0, 50.0)
    except ValueError as exc:
        print(exc)
        return 1

    total_clearance = robot_radius + clearance
    obstacle_map = ObstacleMap(text=args.text, total_clearance=total_clearance)

    try:
        start = ask_state("start", obstacle_map, tuple(args.start) if args.start is not None else None)
        goal = ask_state("goal", obstacle_map, tuple(args.goal) if args.goal is not None else None)
    except ValueError as exc:
        print(exc)
        return 1

    sample_predecessors = list(generate_predecessors(goal, step_size))

    print("=" * 78)
    print("Obstacle map, discretization, and action model")
    print(f"Obstacle text       : {obstacle_map.text}")
    print(f"Robot radius        : {robot_radius}")
    print(f"Clearance           : {clearance}")
    print(f"Total inflation     : {total_clearance}")
    print(f"Step size           : {step_size}")
    print(f"Start index         : {state_to_index(start)}")
    print(f"Goal index          : {state_to_index(goal)}")
    print(f"Sample predecessors : {len(sample_predecessors)}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(run())

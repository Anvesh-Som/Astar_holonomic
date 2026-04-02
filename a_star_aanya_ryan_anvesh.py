#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

MAP_WIDTH = 600
MAP_HEIGHT = 250
THETA_RESOLUTION_DEG = 30
DEFAULT_ROBOT_RADIUS = 5.0
DEFAULT_CLEARANCE = 5.0
DEFAULT_STEP_SIZE = 5.0
DEFAULT_TEXT = "TEAM 1"


@dataclass(frozen=True)
class State:
    x: float
    y: float
    theta_deg: int


def normalize_input_theta(theta_deg: int) -> int:
    if theta_deg % THETA_RESOLUTION_DEG != 0:
        raise ValueError(f"Theta must be a multiple of {THETA_RESOLUTION_DEG} degrees.")
    return theta_deg % 360


def make_state(x: float, y: float, theta_deg: int) -> State:
    return State(float(x), float(y), normalize_input_theta(int(theta_deg)))


def parse_triplet(raw: str) -> Optional[Tuple[float, float, int]]:
    parts = [token for token in raw.replace(",", " ").split() if token]
    if len(parts) != 3:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        theta = int(parts[2])
        return (x, y, theta)
    except ValueError:
        return None


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
    parser = argparse.ArgumentParser(description="Backward A* scaffold for ENPM661 Project 03 - Phase 1")
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

    start = None
    goal = None

    try:
        if args.start is not None:
            start = make_state(*args.start)
        if args.goal is not None:
            goal = make_state(*args.goal)
    except ValueError as exc:
        print(exc)
        return 1

    print("=" * 78)
    print("Backward A* project scaffold")
    print(f"Workspace           : {MAP_WIDTH} x {MAP_HEIGHT}")
    print(f"Obstacle text       : {args.text}")
    print(f"Robot radius        : {robot_radius}")
    print(f"Clearance           : {clearance}")
    print(f"Step size           : {step_size}")
    if start is not None:
        print(f"Start               : ({start.x:.2f}, {start.y:.2f}, {start.theta_deg})")
    if goal is not None:
        print(f"Goal                : ({goal.x:.2f}, {goal.y:.2f}, {goal.theta_deg})")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(run())

# ENPM661 Project 03 - Phase 1
## Backward A* for a Mobile Robot with Holonomic constraints

This submission contains a backward A* implementation for the Project 03 Phase 1 motion-planning problem.

Team-
Aanya Loomba 122298880
Ryan Lowe 122258389
Anvesh Som 122298682

## Included files
- `a_star_aanya_ryan_anvesh.py` - source code
- `output.mp4` - example animation showing exploration and final path
- `README.md` - run instructions and implementation notes

## Dependencies
- Python 3.10+
- `numpy`
- `opencv-python`

Install the Python packages with:

```bash
pip install numpy opencv-python
```

## How to run
### Command-line execution

```bash
python3 a_star_aanya_ryan_anvesh.py \
  --start 50 30 0 \
  --goal 550 220 0 \
  --robot-radius 5 \
  --clearance 5 \
  --step-size 10 \
  --video output.mp4
```

### Interactive execution
If `--start` and `--goal` are omitted, the script prompts for user input.

```bash
python3 a_star_aanya_ryan_anvesh.py
```

The script asks for:
- start state: `(x, y, theta)`
- goal state: `(x, y, theta)`
- robot radius
- clearance
- step size

## Input conventions
- Workspace origin is at the **bottom-left** corner.
- Start and goal coordinates are interpreted in the workspace frame.
- Theta is in degrees and must be a multiple of `30`.
- The allowed step-size range is `1 <= L <= 10`.

## Workspace and obstacle model
- Workspace size: `600 x 250`
- Duplicate-node resolution: `0.5` in x/y and `30` degrees in theta
- Goal threshold: `1.5` units (assignment recommendation)
- Effective obstacle inflation: `robot radius + clearance`

The obstacle map is generated analytically by reusing the Project 02 map-building style and scaling it into the Project 03 workspace. The map includes:
- workspace walls
- half-plane style rectangular primitives
- semi-algebraic rounded glyphs using ellipse/ring models

## Implementation notes
- Search direction: **backward A*** (`goal -> start`)
- Heuristic: Euclidean distance to the start position
- Motion model: 5 actions using orientation changes `{0, +30, +60, -30, -60}` with one forward step of length `L`
- Duplicate handling uses the visited matrix size recommended in the assignment: `500 x 1200 x 12`
- The animation is generated only after the search completes, so exploration is shown first and the optimal path is highlighted afterward

## Sample configuration used for the attached video
```text
Start  : (50, 30, 0)
Goal   : (550, 220, 0)
Radius : 5
Clearance : 5
Step size : 10
```


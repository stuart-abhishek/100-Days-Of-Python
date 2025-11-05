# Day 10 — Sudoku Engineer 🧩
# Author: Stuart Abhishek
#
# Features:
# - Solve Sudoku via Exact Cover (Algorithm X)
# - Generate Sudoku with guaranteed unique solution
# - Difficulty rating from search stats
# - Clean CLI: solve from string/file, or generate by difficulty
#
# Board format:
#   9 lines of 9 characters (digits 1-9 or . for blank), or a single 81-char string.

from __future__ import annotations
import random
import sys
import time
from typing import Dict, List, Set, Tuple, Optional

# ---------------------------
# Exact Cover modeling (base)
# ---------------------------
# We encode each candidate (r,c,d) as satisfying 4 constraints:
# 1) Cell constraint: each cell has exactly one digit        (81)
# 2) Row constraint: each row has each digit once            (81)
# 3) Column constraint: each column has each digit once      (81)
# 4) Box constraint: each 3x3 box has each digit once        (81)
# -> 324 columns total. Each candidate is a row covering 4 columns.

RANGE9 = range(9)

def box_id(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)

def encode_columns() -> Dict[str, int]:
    """Map each constraint key to a column index [0..323]."""
    col = {}
    idx = 0
    # Cell constraints: C(r,c)
    for r in RANGE9:
        for c in RANGE9:
            col[f"C({r},{c})"] = idx; idx += 1
    # Row-digit: R(r,d)
    for r in RANGE9:
        for d in RANGE9:
            col[f"R({r},{d})"] = idx; idx += 1
    # Col-digit: K(c,d)
    for c in RANGE9:
        for d in RANGE9:
            col[f"K({c},{d})"] = idx; idx += 1
    # Box-digit: B(b,d)
    for b in RANGE9:
        for d in RANGE9:
            col[f"B({b},{d})"] = idx; idx += 1
    return col  # size 324

COL_INDEX = encode_columns()

def candidate_columns(r: int, c: int, d: int) -> List[int]:
    """Columns covered by placing digit d (0..8 => 1..9) at (r,c)."""
    return [
        COL_INDEX[f"C({r},{c})"],
        COL_INDEX[f"R({r},{d})"],
        COL_INDEX[f"K({c},{d})"],
        COL_INDEX[f"B({box_id(r,c)},{d})"],
    ]

# ---------------------------
# Algorithm X (set-based)
# ---------------------------

class ExactCover:
    """
    Minimal set-based Algorithm X (not DLX) for clarity.
    universe: 0..m-1 columns
    rows: id -> set(columns it covers)
    """
    def __init__(self, m: int):
        self.m = m
        self.rows: Dict[int, Set[int]] = {}
        self.col_to_rows: List[Set[int]] = [set() for _ in range(m)]

    def add_row(self, row_id: int, cols: List[int]):
        s = set(cols)
        self.rows[row_id] = s
        for c in s:
            self.col_to_rows[c].add(row_id)

    def remove_row(self, row_id: int):
        for c in self.rows[row_id]:
            self.col_to_rows[c].discard(row_id)

    def restore_row(self, row_id: int):
        for c in self.rows[row_id]:
            self.col_to_rows[c].add(row_id)

    def choose_column(self, active_cols: Set[int]) -> int:
        # Heuristic: choose the column with minimum candidate rows (MRV)
        return min(active_cols, key=lambda c: len(self.col_to_rows[c]))

    def search(self, active_cols: Set[int], solution: List[int], stats) -> bool:
        if not active_cols:
            return True
        c = self.choose_column(active_cols)
        rows = list(self.col_to_rows[c])
        stats.nodes += 1
        if not rows:
            return False
        # Try each row that covers column c
        for r in rows:
            removed_cols = []
            removed_rows_per_col: List[Tuple[int, List[int]]] = []
            # Cover row r: remove all columns it covers and conflicting rows
            cols_to_cover = list(self.rows[r])
            for cc in cols_to_cover:
                if cc in active_cols:
                    active_cols.remove(cc)
                    removed_cols.append(cc)
                    # remove all rows that hit cc
                    conflict_rows = list(self.col_to_rows[cc])
                    removed = []
                    for rr in conflict_rows:
                        if rr != r:
                            self.remove_row(rr)
                            removed.append(rr)
                    removed_rows_per_col.append((cc, removed))
            solution.append(r)

            if self.search(active_cols, solution, stats):
                return True

            # Backtrack
            stats.backtracks += 1
            solution.pop()
            # Restore rows
            for cc, removed in reversed(removed_rows_per_col):
                for rr in removed:
                    self.restore_row(rr)
            # Restore cols
            for cc in reversed(removed_cols):
                active_cols.add(cc)

        return False

    def count_solutions(self, active_cols: Set[int], limit: int, stats) -> int:
        """Count up to 'limit' solutions (for uniqueness check)."""
        if not active_cols:
            return 1
        c = self.choose_column(active_cols)
        rows = list(self.col_to_rows[c])
        cnt = 0
        stats.nodes += 1
        for r in rows:
            removed_cols = []
            removed_rows_per_col: List[Tuple[int, List[int]]] = []
            cols_to_cover = list(self.rows[r])
            for cc in cols_to_cover:
                if cc in active_cols:
                    active_cols.remove(cc)
                    removed_cols.append(cc)
                    conflict_rows = list(self.col_to_rows[cc])
                    removed = []
                    for rr in conflict_rows:
                        if rr != r:
                            self.remove_row(rr); removed.append(rr)
                    removed_rows_per_col.append((cc, removed))

            cnt += self.count_solutions(active_cols, limit - cnt, stats)

            # backtrack restore
            for cc, removed in reversed(removed_rows_per_col):
                for rr in removed:
                    self.restore_row(rr)
            for cc in reversed(removed_cols):
                active_cols.add(cc)

            if cnt >= limit:
                break
        return cnt

# ---------------------------
# Sudoku <-> Exact Cover rows
# ---------------------------

def build_ec_for_sudoku(givens: List[List[int]]) -> Tuple[ExactCover, Dict[int, Tuple[int,int,int]]]:
    """
    Build exact cover rows for all valid candidates consistent with givens.
    returns: (EC, row_id -> (r,c,d))
    """
    ec = ExactCover(324)
    id2rcd: Dict[int, Tuple[int,int,int]] = {}
    row_id = 0
    # Precompute existing constraints from givens
    row_used = [[False]*9 for _ in RANGE9]
    col_used = [[False]*9 for _ in RANGE9]
    box_used = [[False]*9 for _ in RANGE9]
    for r in RANGE9:
        for c in RANGE9:
            d = givens[r][c]
            if d != 0:
                d0 = d-1
                if row_used[r][d0] or col_used[c][d0] or box_used[box_id(r,c)][d0]:
                    raise ValueError("Invalid givens (conflict).")
                row_used[r][d0] = col_used[c][d0] = box_used[box_id(r,c)][d0] = True

    # Add rows for all allowed candidates
    for r in RANGE9:
        for c in RANGE9:
            if givens[r][c] != 0:
                d0 = givens[r][c]-1
                cols = candidate_columns(r, c, d0)
                ec.add_row(row_id, cols)
                id2rcd[row_id] = (r, c, d0)
                row_id += 1
            else:
                for d0 in RANGE9:
                    if not (row_used[r][d0] or col_used[c][d0] or box_used[box_id(r,c)][d0]):
                        cols = candidate_columns(r, c, d0)
                        ec.add_row(row_id, cols)
                        id2rcd[row_id] = (r, c, d0)
                        row_id += 1
    return ec, id2rcd

# ---------------------------
# Solver & Utilities
# ---------------------------

class Stats:
    def __init__(self): self.nodes = 0; self.backtracks = 0

def parse_board(s: str) -> List[List[int]]:
    s = "".join(ch for ch in s if ch in "0123456789.")
    if len(s) != 81: raise ValueError("Board must have 81 chars of digits or '.'")
    grid = [[0]*9 for _ in RANGE9]
    for i,ch in enumerate(s):
        r, c = divmod(i, 9)
        grid[r][c] = 0 if ch == '.' or ch == '0' else int(ch)
    return grid

def board_to_string(grid: List[List[int]]) -> str:
    return "".join(str(grid[r][c]) for r in RANGE9 for c in RANGE9)

def render(grid: List[List[int]]) -> str:
    lines = []
    for r in RANGE9:
        row = []
        for c in RANGE9:
            v = grid[r][c]
            row.append(str(v) if v != 0 else ".")
            if c in (2,5): row.append("|")
        lines.append(" ".join(row))
        if r in (2,5): lines.append("------+-------+------")
    return "\n".join(lines)

def solve(grid: List[List[int]], want_all: bool=False, limit: int=2) -> Tuple[List[List[int]], Stats, int]:
    """
    Solve grid. If want_all=True, count solutions up to 'limit' (used for uniqueness).
    Returns: (solution_grid, stats, num_solutions_found)
    """
    ec, id2rcd = build_ec_for_sudoku(grid)
    active_cols = set(range(324))
    stats = Stats()
    sol_rows: List[int] = []
    if want_all:
        count = ec.count_solutions(active_cols, limit, stats)
        # We also need one solution to display if exists:
        # run a search once to get an example solution if unique
        solution_grid = [row[:] for row in grid]
        if count >= 1:
            # run single-solution search to fill solution_grid
            active_cols = set(range(324))
            stats2 = Stats()
            ec2, id2rcd2 = build_ec_for_sudoku(grid)
            sol2: List[int] = []
            ec2.search(active_cols, sol2, stats2)
            for rid in sol2:
                r,c,d0 = id2rcd2[rid]
                solution_grid[r][c] = d0+1
        return solution_grid, stats, count
    else:
        found = ec.search(active_cols, sol_rows, stats)
        solution_grid = [row[:] for row in grid]
        if found:
            for rid in sol_rows:
                r,c,d0 = id2rcd[rid]
                solution_grid[r][c] = d0+1
            return solution_grid, stats, 1
        return solution_grid, stats, 0

# ---------------------------
# Generator (unique puzzle)
# ---------------------------

def complete_grid() -> List[List[int]]:
    """Backtracking fill to make a random complete valid solution grid."""
    grid = [[0]*9 for _ in RANGE9]
    digits = list(range(1,10))

    def safe(r,c,val):
        br, bc = 3*(r//3), 3*(c//3)
        for i in RANGE9:
            if grid[r][i] == val or grid[i][c] == val: return False
        for rr in range(br, br+3):
            for cc in range(bc, bc+3):
                if grid[rr][cc] == val: return False
        return True

    def fill(pos=0):
        if pos == 81: return True
        r, c = divmod(pos, 9)
        if grid[r][c] != 0: return fill(pos+1)
        random.shuffle(digits)
        for v in digits:
            if safe(r,c,v):
                grid[r][c] = v
                if fill(pos+1): return True
                grid[r][c] = 0
        return False

    fill()
    return grid

def generate_unique(target_clues: int = 30, max_attempts: int = 1000) -> Tuple[List[List[int]], dict]:
    """
    Start from a full grid, remove numbers symmetrically while keeping a unique solution.
    target_clues: aim for this many givens (>=17 typically).
    Returns: (puzzle_grid, rating_info)
    """
    full = complete_grid()
    cells = [(r,c) for r in RANGE9 for c in RANGE9]
    pairs = []
    # symmetry pairs (r,c) with (8-r,8-c)
    seen = set()
    for r,c in cells:
        if (r,c) in seen: continue
        rr, cc = 8-r, 8-c
        pairs.append(((r,c), (rr,cc)))
        seen.add((r,c)); seen.add((rr,cc))

    random.shuffle(pairs)
    puzzle = [row[:] for row in full]

    # remove pairs while uniqueness holds and clues >= target
    clues = 81
    for (r1,c1),(r2,c2) in pairs:
        if clues <= target_clues:
            break
        saved1, saved2 = puzzle[r1][c1], puzzle[r2][c2]
        puzzle[r1][c1] = puzzle[r2][c2] = 0
        # uniqueness check: at most 1 solution
        _, stats_check, count = solve(puzzle, want_all=True, limit=2)
        if count != 1:
            # revert removal
            puzzle[r1][c1] = saved1
            puzzle[r2][c2] = saved2
        else:
            clues -= 2

    # difficulty rating: solve once and collect stats
    solved, stats, count = solve(puzzle, want_all=False)
    rating = difficulty_from_stats(stats)
    info = {"clues": sum(1 for r in RANGE9 for c in RANGE9 if puzzle[r][c]!=0),
            "nodes": stats.nodes, "backtracks": stats.backtracks, "rating": rating}
    return puzzle, info

def difficulty_from_stats(stats: Stats) -> str:
    # Crude heuristic based on search effort
    n, b = stats.nodes, stats.backtracks
    if n < 400 and b < 50: return "Easy"
    if n < 2000 and b < 400: return "Medium"
    if n < 7000 and b < 1800: return "Hard"
    return "Expert"

# ---------------------------
# CLI
# ---------------------------

def read_board_from_file(path: str) -> List[List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    s = "".join(ch for line in lines for ch in line if ch in "0123456789.")
    return parse_board(s)

def main():
    print("🧩 Day 10 — Sudoku Engineer (Algorithm X)")
    print("1) Solve from 81-char string")
    print("2) Solve from text file")
    print("3) Generate puzzle (unique) by target clues")
    print("q) Quit")
    choice = input("Choose: ").strip().lower()
    if choice == "1":
        s = input("Enter 81-char puzzle (digits/.) : ").strip()
        grid = parse_board(s)
        print("\nInput:\n" + render(grid))
        t0 = time.time()
        solved, stats, count = solve(grid)
        dt = time.time() - t0
        if count == 1:
            print("\n✅ Solved:")
            print(render(solved))
            print(f"Time: {dt:.3f}s | Nodes: {stats.nodes} | Backtracks: {stats.backtracks}")
        else:
            print("\n❌ No solution.")
    elif choice == "2":
        path = input("Path to puzzle file: ").strip()
        grid = read_board_from_file(path)
        print("\nInput:\n" + render(grid))
        t0 = time.time()
        solved, stats, count = solve(grid)
        dt = time.time() - t0
        if count == 1:
            print("\n✅ Solved:")
            print(render(solved))
            print(f"Time: {dt:.3f}s | Nodes: {stats.nodes} | Backtracks: {stats.backtracks}")
        else:
            print("\n❌ No solution.")
    elif choice == "3":
        target = input("Target number of clues (17..40)? [default 28]: ").strip()
        target = int(target) if target else 28
        t0 = time.time()
        puzzle, info = generate_unique(target_clues=max(17, min(60, target)))
        dt = time.time() - t0
        print("\n🧩 Generated puzzle:")
        print(render(puzzle))
        print(f"\nClues: {info['clues']} | Rating: {info['rating']}")
        print(f"Generator effort — Nodes: {info['nodes']} | Backtracks: {info['backtracks']} | Time: {dt:.2f}s")
        print("\nTip: Copy puzzle as 81 chars (row-major, '.' for blanks):")
        print(board_to_string(puzzle))
    elif choice == "q":
        print("Bye!")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
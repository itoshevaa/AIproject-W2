"""Maxn search agent for the 4-player 5x5 Halma variant."""
from __future__ import annotations

import itertools
import random
import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

from halma import initial_pos, win_cells_all

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TEAM_NUMBER: int = 24
PLAYERS: Tuple[int, int, int, int] = (1, 2, 3, 4)
BOARD_N: int = 5

SEARCH_DEPTH_PLIES: int = 8
TIME_BUDGET_SECONDS: float = 1.2

ASSUMED_MOVE_LIMIT: int = 100

GOAL_CELLS: Dict[int, Tuple[Tuple[int, int], ...]] = {
    p: tuple(win_cells_all[p]) for p in PLAYERS
}
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_INITIAL_SIGNATURE = tuple(tuple(row) for row in initial_pos)

Cell = Tuple[int, int]
State = Tuple[FrozenSet[Cell], FrozenSet[Cell], FrozenSet[Cell], FrozenSet[Cell]]
Move = Tuple[Cell, Cell]

# --------------------------------------------------------------------------
# Board <-> state conversion
# --------------------------------------------------------------------------

def board_to_state(board: List[List[int]]) -> State:
    piles: Tuple[List[Cell], List[Cell], List[Cell], List[Cell]] = ([], [], [], [])
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            value = board[r][c]
            if value:
                piles[value - 1].append((r, c))
    return tuple(frozenset(p) for p in piles)  # type: ignore[return-value]


def occupied_cells(state: State) -> FrozenSet[Cell]:
    occ: set = set()
    for cells in state:
        occ |= cells
    return frozenset(occ)


def pos_to_label(pos: Cell) -> str:
    row, col = pos
    return f"{chr(ord('A') + col)}{row + 1}"


# --------------------------------------------------------------------------
# Move generation
# --------------------------------------------------------------------------

def legal_moves_for(state: State, player: int) -> List[Move]:
    occ = occupied_cells(state)
    moves: List[Move] = []
    for pos in state[player - 1]:
        r, c = pos
        for dr, dc in DIRECTIONS:
            step = (r + dr, c + dc)
            if 0 <= step[0] < BOARD_N and 0 <= step[1] < BOARD_N and step not in occ:
                moves.append((pos, step))
            mid = (r + dr, c + dc)
            land = (r + 2 * dr, c + 2 * dc)
            if (
                0 <= land[0] < BOARD_N
                and 0 <= land[1] < BOARD_N
                and mid in occ
                and land not in occ
            ):
                moves.append((pos, land))
    return moves


def apply_move(state: State, player: int, old_pos: Cell, new_pos: Cell) -> State:
    idx = player - 1
    new_cells = frozenset((state[idx] - {old_pos}) | {new_pos})
    return state[:idx] + (new_cells,) + state[idx + 1:]


def check_winner(state: State) -> Optional[int]:
    for player in PLAYERS:
        cells = state[player - 1]
        if len(cells) == 3 and all(pos in GOAL_CELLS[player] for pos in cells):
            return player
    return None


# --------------------------------------------------------------------------
# Distance field / evaluation
# --------------------------------------------------------------------------

def bfs_distance(start: Cell, obstacles: FrozenSet[Cell]) -> Dict[Cell, int]:
    dist: Dict[Cell, int] = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        d = dist[(r, c)]
        for dr, dc in DIRECTIONS:
            step = (r + dr, c + dc)
            if (
                0 <= step[0] < BOARD_N
                and 0 <= step[1] < BOARD_N
                and step not in obstacles
                and step not in dist
            ):
                dist[step] = d + 1
                queue.append(step)
            mid = (r + dr, c + dc)
            land = (r + 2 * dr, c + 2 * dc)
            if (
                0 <= land[0] < BOARD_N
                and 0 <= land[1] < BOARD_N
                and mid in obstacles
                and land not in obstacles
                and land not in dist
            ):
                dist[land] = d + 1
                queue.append(land)
    return dist


_UNREACHABLE_PENALTY = 50  # bigger than any real distance on a 5x5 board


def _piece_to_goal_distance(piece: Cell, goal: Cell, occ: FrozenSet[Cell]) -> int:
    # Goal cells may currently hold opponent pieces, so omit the target
    # itself while keeping intermediate blockers available for jumps.
    obstacles = occ - {piece, goal}
    dist = bfs_distance(piece, obstacles)
    return dist.get(goal, _UNREACHABLE_PENALTY)


def assignment_distance(state: State, player: int) -> int:
    pieces = list(state[player - 1])
    goals = GOAL_CELLS[player]
    if not pieces:
        return _UNREACHABLE_PENALTY * 3
    occ = occupied_cells(state)
    pairwise = {
        (piece, goal): _piece_to_goal_distance(piece, goal, occ)
        for piece in pieces
        for goal in goals
    }

    best = None
    for perm in itertools.permutations(goals, len(pieces)):
        total = sum(pairwise[(pieces[i], goal)] for i, goal in enumerate(perm))
        if best is None or total < best:
            best = total
    return best if best is not None else _UNREACHABLE_PENALTY * 3


_MAX_REASONABLE_DIST = 24.0  # normalisation constant, generous for a 5x5 board


def evaluate(state: State, turn_estimate: Optional[int], move_limit: int) -> Dict[int, float]:
    winner = check_winner(state)
    if winner is not None:
        return {p: (1.0 if p == winner else 0.0) for p in PLAYERS}

    if turn_estimate is not None:
        urgency = (turn_estimate - 0.7 * move_limit) / (0.3 * move_limit + 1e-9)
        urgency = min(1.0, max(0.0, urgency))
    else:
        urgency = 0.0

    scores: Dict[int, float] = {}
    for player in PLAYERS:
        dist = assignment_distance(state, player)
        progress = max(0.0, 1.0 - dist / _MAX_REASONABLE_DIST)
        on_goal = sum(1 for pos in state[player - 1] if pos in GOAL_CELLS[player])
        goal_bonus = on_goal / 3.0
        score = 0.6 * progress + 0.4 * goal_bonus

        if urgency > 0.0:
            occupying_others = sum(
                1
                for other in PLAYERS
                if other != player
                for pos in state[player - 1]
                if pos in GOAL_CELLS[other]
            )
            if occupying_others:
                score -= urgency * 0.5 * min(1.0, occupying_others / 3.0)

        scores[player] = min(1.0, max(0.0, score))
    return scores


def _one_ply_gain(state: State, player: int, move: Move) -> int:
    old_pos, new_pos = move
    before = assignment_distance(state, player)
    after = assignment_distance(apply_move(state, player, old_pos, new_pos), player)
    return before - after


def order_moves(state: State, player: int, moves: List[Move]) -> List[Move]:
    return sorted(moves, key=lambda m: _one_ply_gain(state, player, m), reverse=True)


# ---------------
# Maxn search
# ---------------

class SearchStats:
    __slots__ = ("nodes", "tt_hits", "pruned_branches")

    def __init__(self) -> None:
        self.nodes = 0
        self.tt_hits = 0
        self.pruned_branches = 0


TT: Dict[Tuple[State, int, int], Dict[int, float]] = {}


def maxn(
    state: State,
    mover: int,
    depth: int,
    turn_estimate: Optional[int],
    move_limit: int,
    stats: SearchStats,
    path_states: FrozenSet[State] = frozenset(),
    deadline: Optional[float] = None,
) -> Tuple[Dict[int, float], Optional[Move]]:
    stats.nodes += 1

    winner = check_winner(state)
    if winner is not None:
        return {p: (1.0 if p == winner else 0.0) for p in PLAYERS}, None

    if depth == 0 or (deadline is not None and time.perf_counter() > deadline):
        return evaluate(state, turn_estimate, move_limit), None

    if state in path_states:
        # Repeated position on this line: treat as a dead end rather than
        # looping forever. Static evaluation is an honest stand-in.
        return evaluate(state, turn_estimate, move_limit), None

    key = (state, mover, depth)
    cached = TT.get(key)
    if cached is not None:
        stats.tt_hits += 1
        return cached, None

    moves = legal_moves_for(state, mover)
    next_mover = mover % 4 + 1
    next_turn_estimate = None if turn_estimate is None else turn_estimate + 1

    if not moves:
        # No legal moves: this seat's turn is skipped, but the game still
        # advances one ply (matches skip_turn/finish_turn in the harness)
        vec, _ = maxn(
            state, next_mover, depth - 1, next_turn_estimate, move_limit,
            stats, path_states, deadline,
        )
        return vec, None

    ordered = order_moves(state, mover, moves)
    new_path = path_states | {state}

    best_vec: Optional[Dict[int, float]] = None
    best_move: Optional[Move] = None

    for i, (old_pos, new_pos) in enumerate(ordered):
        child_state = apply_move(state, mover, old_pos, new_pos)
        child_vec, _ = maxn(
            child_state, next_mover, depth - 1, next_turn_estimate, move_limit,
            stats, new_path, deadline,
        )
        if best_vec is None or child_vec[mover] > best_vec[mover]:
            best_vec = child_vec
            best_move = (old_pos, new_pos)

        # Shallow pruning (sound for maxn, unlike 2-player alpha-beta):
        # once the mover has found the maximum possible score for itself,
        # no remaining sibling can improve on that
        if best_vec[mover] >= 0.999:
            stats.pruned_branches += len(ordered) - i - 1
            break

    assert best_vec is not None and best_move is not None
    TT[key] = best_vec
    return best_vec, best_move


def maxn_root(
    state: State,
    mover: int,
    depth: int,
    turn_estimate: Optional[int],
    move_limit: int,
    stats: SearchStats,
    deadline: Optional[float] = None,
) -> Tuple[List[Tuple[Move, float]], bool]:
    """Return scored root moves and whether the ranking was complete."""
    moves = legal_moves_for(state, mover)
    if not moves:
        return [], True

    ordered = order_moves(state, mover, moves)
    next_mover = mover % 4 + 1
    next_turn_estimate = None if turn_estimate is None else turn_estimate + 1
    path = frozenset({state})

    results: List[Tuple[Move, float]] = []
    best_score = None
    for old_pos, new_pos in ordered:
        child_state = apply_move(state, mover, old_pos, new_pos)
        child_vec, _ = maxn(
            child_state, next_mover, depth - 1, next_turn_estimate, move_limit,
            stats, path, deadline,
        )
        score = child_vec[mover]
        results.append(((old_pos, new_pos), score))
        if best_score is None or score > best_score:
            best_score = score
        if best_score >= 0.999:
            break

        if deadline is not None and time.perf_counter() > deadline:
            return results, False

    return results, True


# ---------------
# Turn tracking
# ---------------

_call_count: Dict[int, int] = {p: 0 for p in PLAYERS}

RECENT_WINDOW = 12
_recent_squares: Dict[int, deque] = {p: deque(maxlen=RECENT_WINDOW) for p in PLAYERS}

# Only near-tied root moves are steered away from recently used squares
REVISIT_SCORE_MARGIN = 0.03


def _update_turn_tracking(board: List[List[int]], player: int) -> int:
    signature = tuple(tuple(row) for row in board)
    if signature == _INITIAL_SIGNATURE:
        for p in PLAYERS:
            _call_count[p] = 0
            _recent_squares[p] = deque(maxlen=RECENT_WINDOW)
    _call_count[player] += 1
    n = _call_count[player]
    return (n - 1) * 4 + (player - 1)


def _select_move_avoiding_revisits(
    results: List[Tuple[Move, float]], player: int
) -> Optional[Move]:
    if not results:
        return None

    ranked = sorted(results, key=lambda r: r[1], reverse=True)
    top_move, top_score = ranked[0]

    if top_score >= 0.999:
        return top_move

    recent = _recent_squares.get(player, ())
    for move, score in ranked:
        if top_score - score > REVISIT_SCORE_MARGIN:
            break
        _, new_pos = move
        if new_pos not in recent:
            return move

    return top_move


# ------------------
# Input validation
# ------------------

def _validate_input(board: List[List[int]], player: int) -> None:
    if not isinstance(board, list) or len(board) != BOARD_N:
        raise ValueError("Board must be a 5x5 list of lists")
    for row in board:
        if not isinstance(row, list) or len(row) != BOARD_N:
            raise ValueError("Board must be a 5x5 list of lists")
        for value in row:
            if value not in (0, 1, 2, 3, 4):
                raise ValueError(f"Invalid cell value: {value!r}")
    if player not in PLAYERS:
        raise ValueError(f"Invalid player: {player!r}")

    counts = {p: sum(row.count(p) for row in board) for p in PLAYERS}
    for p, count in counts.items():
        if count > 3:
            raise ValueError(f"Player {p} has {count} pieces on the board, expected at most 3")
    if counts[player] == 0:
        raise ValueError(f"Player {player} has no pieces on the board")


# ----------------
# Visualization
# ----------------

DISPLAY_PLIES = 4
DISPLAY_BRANCH_CAP = 2


def _build_and_render_tree(state: State, mover: int, turn_estimate: Optional[int]) -> None:
    try:
        from treelib import Tree
    except ImportError:
        print("treelib is not installed (pip install treelib); skipping visualization.")
        return

    tree = Tree()
    tree.create_node(f"Root (Player {mover} to move)", "root")
    counter = itertools.count(1)

    def recurse(cur_state: State, cur_mover: int, plies_left: int, parent_id: str,
                path: FrozenSet[State]) -> None:
        if plies_left == 0 or check_winner(cur_state) is not None:
            return
        all_moves = order_moves(cur_state, cur_mover, legal_moves_for(cur_state, cur_mover))
        if not all_moves:
            return
        moves = all_moves[:DISPLAY_BRANCH_CAP]
        hidden_count = len(all_moves) - len(moves)
        next_mover = cur_mover % 4 + 1
        new_path = path | {cur_state}
        best_for_mover: Optional[float] = None

        for i, (old_pos, new_pos) in enumerate(moves):
            child_state = apply_move(cur_state, cur_mover, old_pos, new_pos)
            node_id = f"n{next(counter)}"
            base_label = f"P{cur_mover}: {pos_to_label(old_pos)}->{pos_to_label(new_pos)}"

            if child_state in new_path:
                tree.create_node(base_label + "  [pruned: repeated position]", node_id, parent=parent_id)
                continue

            child_vec = evaluate(child_state, turn_estimate, ASSUMED_MOVE_LIMIT)
            score_str = " ".join(f"P{p}:{child_vec[p]:.2f}" for p in PLAYERS)
            tree.create_node(f"{base_label}\n{score_str}", node_id, parent=parent_id)

            score = child_vec[cur_mover]
            if best_for_mover is not None and best_for_mover >= 0.999:
                remaining = len(moves) - i
                tree.create_node(f"... {remaining} sibling(s) skipped [shallow pruning]",
                                  f"{node_id}_cut", parent=parent_id)
                break

            recurse(child_state, next_mover, plies_left - 1, node_id, new_path)
            if best_for_mover is None or score > best_for_mover:
                best_for_mover = score

        if hidden_count > 0:
            tree.create_node(f"... {hidden_count} lower-ranked move(s) not shown",
                              f"n{next(counter)}", parent=parent_id)

    recurse(state, mover, DISPLAY_PLIES, "root", frozenset())

    text_path = f"Team{TEAM_NUMBER}_Tree.txt"
    tree.save2file(text_path)
    print(f"Search tree (text) written to {text_path}")

    try:
        dot_path = f"Team{TEAM_NUMBER}_Tree.dot"
        tree.to_graphviz(filename=dot_path, shape="box")
        with open(dot_path, "r", encoding="utf-8") as dot_file:
            dot_source = dot_file.read()
        if not dot_source:
            raise RuntimeError("treelib produced an empty DOT file")
        import graphviz as gv

        styled_source = dot_source.replace(
            "digraph tree {",
            'digraph tree {\n\trankdir=TB; ranksep=0.5; nodesep=0.15; '
            'node [fontsize=9, fontname="Helvetica"];',
            1,
        )
        src = gv.Source(styled_source)
        stem = f"Team{TEAM_NUMBER}_Tree"

        src.format = "png"
        src.render(filename=stem, cleanup=True)
        print(f"Search tree image written to {stem}.png "
              f"(required by the assignment brief)")

        src.format = "pdf"
        src.render(filename=stem, cleanup=True)
        print(f"Search tree also written to {stem}.pdf (easier to open/print)")
    except Exception as error:
        print(f"Could not render PNG/PDF ({type(error).__name__}: {error}). "
              f"Text tree above is the fallback; install graphviz "
              f"(pip install graphviz) and the system 'dot' binary for images.")


def _safe_visualize(state: State, mover: int, turn_estimate: Optional[int]) -> None:
    try:
        _build_and_render_tree(state, mover, turn_estimate)
    except Exception as error:
        print(f"Visualization failed ({type(error).__name__}: {error}); continuing without it.")


# --------------------
# Public entry point
# --------------------

def AI_Player_Team24(
    board: List[List[int]], player: int, visualize_tree: bool
) -> Tuple[str, str]:
    try:
        _validate_input(board, player)
        state = board_to_state(board)
        turn_estimate = _update_turn_tracking(board, player)

        TT.clear()
        stats = SearchStats()
        deadline = time.perf_counter() + TIME_BUDGET_SECONDS

        results, _ = maxn_root(
            state, player, 2, turn_estimate, ASSUMED_MOVE_LIMIT,
            stats, deadline=None,
        )

        depth = 4
        while depth <= SEARCH_DEPTH_PLIES and time.perf_counter() < deadline:
            depth_results, exhausted = maxn_root(
                state, player, depth, turn_estimate, ASSUMED_MOVE_LIMIT,
                stats, deadline=deadline,
            )
            if exhausted and depth_results:
                results = depth_results
            else:
                break
            depth += 4

        best_move = _select_move_avoiding_revisits(results, player)

        if best_move is None:
            fallback_moves = legal_moves_for(state, player)
            if not fallback_moves:
                raise ValueError(f"Player {player} has no legal moves")
            best_move = random.choice(fallback_moves)

        if visualize_tree:
            _safe_visualize(state, player, turn_estimate)

        _recent_squares[player].append(best_move[1])
        old_pos, new_pos = best_move
        return pos_to_label(old_pos), pos_to_label(new_pos)

    except Exception as error:
        print(f"[AI_Player_Team{TEAM_NUMBER}] error, using fallback: "
              f"{type(error).__name__}: {error}")
        try:
            state = board_to_state(board)
            fallback_moves = legal_moves_for(state, player)
            if fallback_moves:
                old_pos, new_pos = random.choice(fallback_moves)
                return pos_to_label(old_pos), pos_to_label(new_pos)
        except Exception:
            pass
        raise

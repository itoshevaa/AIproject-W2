"""
AI_Player_Team24.py

Maxn search agent for 4-player Halma (5x5 simplified variant).

Design summary
--------------
- Board state: 4 frozensets of (row, col) cells, one per player. Hashable,
  so it can be used directly as a transposition-table / path key.
- Evaluation: for each player, an exact jump-aware shortest-path distance
  from each of their pieces to an optimal assignment of the 3 goal cells
  (BFS over the 5x5 grid where a "jump over an occupied square" counts as
  a single step, same as a plain move). This is recomputed at every node
  from that node's own occupancy, so tactical resources (a stone to jump
  over) are only rewarded when the tree shows they're actually still
  there when it matters -- not as a static "adjacency bonus", which would
  be wrong half the time in a 4-player alternating-turn game (see notes
  at the bottom of this file).
- Search: maxn (Luckhardt & Irani) to a configurable ply depth, with:
    * move ordering by 1-ply heuristic gain
    * a transposition table (state, mover, depth) -> score vector
    * in-path repetition pruning (avoid re-exploring a state already
      seen earlier on the same root-to-node path)
    * shallow pruning: once a mover has found a branch scoring the
      maximum possible value for itself, remaining siblings cannot
      improve on that, so they're skipped (this is the form of pruning
      that is actually sound for maxn -- classic two-player alpha-beta
      is NOT sound for n > 2 independent players, see Korf 1991)
    * iterative deepening under a wall-clock time budget, so the bot
      always returns a move even if full-depth search doesn't finish
- Turn tracking: the harness never passes move_count to this function.
  But turn order is always a strict 1->2->3->4->1... rotation regardless
  of skipped/illegal turns (see halma_pygame_4players.py: finish_turn is
  called even when a move is illegal, and current_player always advances
  by exactly one seat). So this file tracks how many times *it* has been
  called since the board last matched the initial position, and derives
  the exact global move_count from that. This is used only to know when
  the move limit is approaching, to avoid ending the game standing in
  someone else's end zone (a loss under the move-limit rule).
"""
from __future__ import annotations

import itertools
import random
import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

from halma import check_legal_move, initial_pos, parse_position, win_cells_all

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TEAM_NUMBER: int = 24
PLAYERS: Tuple[int, int, int, int] = (1, 2, 3, 4)
BOARD_N: int = 5

# "Depth of 2 (looking 2 moves ahead for your team)" is read here as two
# full rounds of the table (you, then the other three, twice) = 8 plies.
SEARCH_DEPTH_PLIES: int = 8
TIME_BUDGET_SECONDS: float = 1.2  # was 0.9; widened for headroom on slower
                                   # machines / when pygame's own loop is
                                   # competing for CPU during a live game

# The provided harness (halma_pygame_4players.py) uses 100; not passed to
# us, so it's assumed here. Change this if your grading harness differs.
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
#
# bfs_distance treats ONE piece as the mover and every other occupied cell
# (from any player) as a fixed obstacle that can be stepped next to or
# jumped over. It is recomputed per node, per piece, so it is always an
# accurate snapshot of "how many moves to goal right now" -- it naturally
# rewards jump chains without needing a separate static adjacency bonus.
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
    # The goal cell itself is excluded from the obstacle set: at the start
    # of the game your goal cells are literally the opponent's home
    # squares, so treating "currently occupied" as "permanently blocked"
    # would make the heuristic blind for the whole opening. Whoever is
    # sitting there is essentially certain to have moved on by the time a
    # piece actually arrives, so distance-to-goal should measure geometry,
    # not that transient occupancy. Cells in between are still real
    # obstacles (and useful ones, for jumping).
    obstacles = occ - {piece, goal}
    dist = bfs_distance(piece, obstacles)
    return dist.get(goal, _UNREACHABLE_PENALTY)


def assignment_distance(state: State, player: int) -> int:
    """Minimum total moves to get this player's 3 pieces onto their 3 goal
    cells, choosing the best pairing of piece -> goal cell."""
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


# --------------------------------------------------------------------------
# Maxn search
# --------------------------------------------------------------------------

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
        # advances one ply (matches skip_turn/finish_turn in the harness).
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
        # no remaining sibling can improve on that.
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
    """Like maxn, but for the root position only: returns every candidate
    move together with its score, instead of just the single best one.
    Needed so the caller can apply anti-oscillation tie-breaking (see
    _select_move_avoiding_revisits) using the runner-up, not just the
    winner -- shallow pruning inside maxn() itself intentionally throws
    runner-ups away once a perfect score is found, which is fine deeper in
    the tree but would hide the information we need right here at depth 0.

    Also returns `exhausted`: True if every legal move was actually
    evaluated (or the loop stopped only because a mover-maximum score was
    found, which is a sound reason to stop, not a compromise). False if
    the loop was cut short by the deadline -- meaning some legal moves,
    possibly including the true best one, were never looked at, and the
    scores returned are not a trustworthy full ranking. On a slower
    machine (or one under more CPU contention, e.g. from pygame's own
    loop running at the same time), this can happen even at a shallow
    depth; the caller must not treat a non-exhausted result as final.
    """
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
            break  # same sound shallow-pruning rule as inside maxn()

        if deadline is not None and time.perf_counter() > deadline:
            return results, False  # ran out of time -- NOT a full ranking

    return results, True


# --------------------------------------------------------------------------
# Turn tracking (see module docstring)
# --------------------------------------------------------------------------

_call_count: Dict[int, int] = {p: 0 for p in PLAYERS}

# Each player's own last RECENT_WINDOW destination squares, most recent
# last. A short recent history, not "every square ever used" -- the
# bug we're guarding against is short-cycle repetition (going back to a
# square within the last several of our own turns), not legitimate reuse
# of ground later in the game. An unbounded "ever visited" version of
# this was tested and measurably made play worse: it started avoiding
# perfectly good moves just because a piece had passed through that
# square once, much earlier, for an unrelated reason.
from collections import deque as _deque
RECENT_WINDOW = 12
_recent_squares: Dict[int, "_deque"] = {p: _deque(maxlen=RECENT_WINDOW) for p in PLAYERS}

# How close a revisiting move's score must be to the true best move's
# score for us to still prefer a fresh alternative instead. Small on
# purpose: this only breaks near-ties in favour of new ground, it never
# overrides a search result that's genuinely better.
REVISIT_SCORE_MARGIN = 0.03


def _update_turn_tracking(board: List[List[int]], player: int) -> int:
    signature = tuple(tuple(row) for row in board)
    if signature == _INITIAL_SIGNATURE:
        for p in PLAYERS:
            _call_count[p] = 0
            _recent_squares[p] = _deque(maxlen=RECENT_WINDOW)
    _call_count[player] += 1
    n = _call_count[player]
    # Turn order is a strict 1,2,3,4,1,2,... rotation, so this is exact,
    # not an estimate, as long as the game began from the initial board.
    return (n - 1) * 4 + (player - 1)


def _select_move_avoiding_revisits(
    results: List[Tuple[Move, float]], player: int
) -> Optional[Move]:
    """Picks the best root move, but among moves scoring within
    REVISIT_SCORE_MARGIN of the true best, prefers one that doesn't land
    on a square this player has used within its last RECENT_WINDOW own
    moves.

    This targets short-cycle repetition specifically: a fresh search has
    no memory beyond the current board, so if two or three squares keep
    scoring near-identically as the bot shuffles pieces around, it can
    end up re-treading the same route several turns apart (C2->C3 now,
    C3->D3 later, then both again several turns after that) without ever
    technically reversing its immediately previous move. A move that's
    genuinely better than the alternatives (outside the margin) is always
    still taken -- this only breaks ties, and only against *recent*
    ground, so a legitimate return trip through a square used long ago is
    never blocked."""
    if not results:
        return None

    ranked = sorted(results, key=lambda r: r[1], reverse=True)
    top_move, top_score = ranked[0]

    if top_score >= 0.999:
        return top_move  # a forced win: take it, no cleverness needed

    recent = _recent_squares.get(player, ())
    for move, score in ranked:
        if top_score - score > REVISIT_SCORE_MARGIN:
            break  # remaining candidates are meaningfully worse; stop
        _, new_pos = move
        if new_pos not in recent:
            return move

    return top_move  # every near-tied option revisits recent ground


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Visualization (only run when visualize_tree=True; kept separate from the
# move-choosing search since a full 8-ply tree has far too many nodes to
# render as an image -- it's illustrative, capped at a shallow display depth,
# and explicitly tags pruned/repeated branches so the pruning strategy is
# visible.)
# --------------------------------------------------------------------------

DISPLAY_PLIES = 4  # covers one full round (all 4 players) - deeper than a single exchange, still readable
DISPLAY_BRANCH_CAP = 2  # a genuinely full 8-ply, full-branching tree has billions of nodes and cannot be rendered; this stays small enough to read as a PNG (the PDF has no pixel-size limit, if you want a richer capture)
                         # readable, representative slice: deeper than
                         # before, still small enough to actually read
                         # as a PNG (the PDF has no such size limit).


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
        # treelib.to_graphviz only *returns* a string when filename is None,
        # in which case it also prints it to stdout as a side effect (noisy
        # but harmless); writing to a real file and reading it back avoids
        # that side effect and is consistent across treelib versions.
        dot_path = f"Team{TEAM_NUMBER}_Tree.dot"
        tree.to_graphviz(filename=dot_path, shape="box")
        with open(dot_path, "r", encoding="utf-8") as dot_file:
            dot_source = dot_file.read()
        if not dot_source:
            raise RuntimeError("treelib produced an empty DOT file")
        import graphviz as gv
        # (stem computed below, no longer defined as png_stem here)
        # Inject compact layout attributes directly into the DOT source
        # (graphviz.Source has no graph_attr setter; only Digraph does).
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
    except Exception as error:  # visualization must never break gameplay
        print(f"Visualization failed ({type(error).__name__}: {error}); continuing without it.")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def AI_Player_Team24(
    board: List[List[int]], player: int, visualize_tree: bool
) -> Tuple[str, str]:
    try:
        _validate_input(board, player)
        state = board_to_state(board)
        turn_estimate = _update_turn_tracking(board, player)

        TT.clear()  # bounded memory: fresh table each call, own move only
        stats = SearchStats()
        deadline = time.perf_counter() + TIME_BUDGET_SECONDS

        # Guaranteed safety net: a 2-ply (own move + immediate replies)
        # pass, run with NO deadline. This is cheap enough to always finish
        # quickly even on a slow machine, and it means `results` is never
        # empty by the time the deeper, time-limited passes below run --
        # so even if every deeper pass times out mid-evaluation, we still
        # have a complete, trustworthy ranking to fall back on, rather
        # than risking a partial/truncated ranking whose "best" move is
        # only best among the few candidates that happened to be checked
        # first (see the E5<->E3 bug this was fixing: on a slower machine,
        # a deadline could be hit after evaluating only 1-2 of 6+ root
        # candidates, silently biasing the choice toward whatever was
        # ordered first rather than what was actually best).
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
                # Only trust a FULL ranking to replace the previous, also
                # complete, ranking. A deadline-truncated partial result
                # is discarded rather than allowed to override.
                results = depth_results
            else:
                break  # deeper search ran out of time; deeper still would too
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
        # Never raise out of this function: an exception forfeits the turn
        # in the 4-player harness, but a legal fallback move is free points.
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

from collections import deque
from typing import List, Tuple, Set, Optional
from treelib import Tree

# Import rules and setups directly from halma.py
from halma import check_legal_move, initial_pos_1v1, win_cells_1v1

# Target goal squares: {(3, 4), (4, 3), (4, 4)} -> (E4, D5, E5)
GOAL_CELLS: Set[Tuple[int, int]] = set(win_cells_1v1[1])


def pos_to_label(pos: Tuple[int, int]) -> str:
    """Converts (row, col) coordinates to Halma notation (e.g., (0, 0) -> 'A1')."""
    return f"{chr(ord('A') + pos[1])}{pos[0] + 1}"


def state_from_board(board: List[List[int]]) -> Tuple[Tuple[int, int], ...]:
    """Extracts positions of Player 1 pieces into a sorted, hashable tuple."""
    pieces = []
    for r in range(5):
        for c in range(5):
            if board[r][c] == 1:
                pieces.append((r, c))
    return tuple(sorted(pieces))


def board_from_state(state: Tuple[Tuple[int, int], ...]) -> List[List[int]]:
    """Reconstructs a 5x5 matrix from player positions for check_legal_move."""
    board = [[0] * 5 for _ in range(5)]
    for r, c in state:
        board[r][c] = 1
    return board


def heuristic_score(old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> int:
    """
    Branch ordering heuristic: prioritizes moves that get closest to (4, 4).
    Calculates the reduction in Manhattan distance to the bottom-right corner.
    """
    old_dist = abs(4 - old_pos[0]) + abs(4 - old_pos[1])
    new_dist = abs(4 - new_pos[0]) + abs(4 - new_pos[1])
    return old_dist - new_dist  # Higher value = bigger progress towards target


def get_ordered_moves(state: Tuple[Tuple[int, int], ...]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Generates and sorts legal moves to prioritize forward progress."""
    board = board_from_state(state)
    legal_moves = []

    for old_pos in state:
        for r in range(5):
            for c in range(5):
                new_pos = (r, c)
                if check_legal_move(board, old_pos, new_pos):
                    legal_moves.append((old_pos, new_pos))

    # Branch Ordering: sort descending by Manhattan progress
    legal_moves.sort(key=lambda m: heuristic_score(m[0], m[1]), reverse=True)
    return legal_moves


def apply_move(state: Tuple[Tuple[int, int], ...], old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
    """Applies a move and returns a new sorted state tuple."""
    new_pieces = [pos for pos in state if pos != old_pos] + [new_pos]
    return tuple(sorted(new_pieces))


def build_search_tree():
    # 1. Initialize start state and tree
    start_state = state_from_board(initial_pos_1v1)
    tree = Tree()
    tree.create_node("Root (Start)", "root")

    # Queue format: (current_state, node_id, depth)
    queue = deque([(start_state, "root", 0)])

    # Pruning: Set of visited states to prune cycles and duplicates
    visited_states: Set[Tuple[Tuple[int, int], ...]] = {start_state}

    # Tracking parent relationships to highlight shortest path later
    node_parent_map = {}
    goal_node_id: Optional[str] = None
    node_counter = 0

    print("Searching for shortest path and generating tree...")

    # 2. Breadth-First Search (guarantees shortest path)
    while queue:
        current_state, parent_id, depth = queue.popleft()

        # Check if all 3 pieces are in the goal corner
        if set(current_state) == GOAL_CELLS:
            goal_node_id = parent_id
            print(f"Goal reached at depth {depth}!")
            break

        # Generate sorted legal successor moves
        for old_pos, new_pos in get_ordered_moves(current_state):
            next_state = apply_move(current_state, old_pos, new_pos)

            # Implement Pruning: skip duplicate positions
            if next_state not in visited_states:
                visited_states.add(next_state)
                node_counter += 1
                child_id = f"node_{node_counter}"

                move_label = f"{pos_to_label(old_pos)}->{pos_to_label(new_pos)}"
                tree.create_node(move_label, child_id, parent=parent_id)
                node_parent_map[child_id] = (parent_id, move_label)

                queue.append((next_state, child_id, depth + 1))

    # 3. Highlight the shortest path through the tree
    if goal_node_id:
        curr = goal_node_id
        while curr in node_parent_map:
            p_id, lbl = node_parent_map[curr]
            tree.get_node(curr).tag = f"[*SHORTEST PATH*] {lbl}"
            curr = p_id

    # 4. Display the tree in console
    print("\nSearch Tree:")
    tree.show()


if __name__ == "__main__":
    build_search_tree()
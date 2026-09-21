from treelib import Tree


BOARD_SIZE = 5
COLUMNS = "ABCDE"



def coords_in_pos(coords):
    row, col = coords
    return COLUMNS[col] + str(row + 1)


def validity_input(board, player):
    if player not in (1, 2, 3, 4):
        raise ValueError("player must be 1, 2, 3 or 4")
    if len(board) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board):
        raise ValueError("board must be 5x5")
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            square = board[row][col]
            if square == 0:
                continue
            if square not in counts:
                raise ValueError(f"invalid value {square} at {coords_in_pos((row, col))}")
            counts[square] += 1
    if any(n != 3 for n in counts.values()):
        raise ValueError("each player must have 3 pieces")



directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def on_board(coords):
    row, col = coords
    return row in range(BOARD_SIZE) and col in range(BOARD_SIZE)


def get_moves(mine, taken):
    moves = []
    for row, col in mine:
        for directionrow, directioncol in directions: 
            move = (row + directionrow, col + directioncol)
            jump = (row + 2 * directionrow, col + 2 * directioncol)
            if on_board(move) and move not in taken:
                moves.append(((row, col), move))
            elif on_board(jump) and move in taken and jump not in taken:
                moves.append(((row, col), jump))  
    return moves


def distance_to_goal(player, pieces):
    if player == 1:
        goal_row, goal_col = 4, 4     
    elif player == 2:
        goal_row, goal_col = 4, 0   
    elif player == 3:
        goal_row, goal_col = 0, 0    
    else:
        goal_row, goal_col = 0, 4    
    return sum(abs(row - goal_row) + abs(col - goal_col) for row, col in pieces) - 2

def score(all_pieces):
    return tuple(20-distance_to_goal(player, all_pieces[player-1]) for player in range(1,5))


def replace_pieces(all_pieces, player, new_pieces):
    p1, p2, p3, p4 = all_pieces
    if player == 1:
        p1 = new_pieces
    elif player == 2:
        p2 = new_pieces
    elif player == 3:
        p3 = new_pieces
    else:
        p4 = new_pieces
    return (p1, p2, p3, p4)


def maxn(all_pieces, player, turns, bound, table, tree, parent):
    if turns == 5:                         
        return score(all_pieces), None
 
    key = (all_pieces, player, turns)
    if key in table:                     
        return table[key]
 
    occupied = set()
    for pieces in all_pieces:
        occupied.update(pieces) 

    children = []

    for old, new in get_moves(all_pieces[player - 1], occupied):
        new_pieces = []
        for square in all_pieces[player - 1]:
            if square == old:
                new_pieces.append(new)       
            else:
                new_pieces.append(square)     
        new_pieces.sort()
        new_pieces = tuple(new_pieces)        
        children.append((distance_to_goal(player, new_pieces), old, new, replace_pieces(all_pieces, player, new_pieces)))

    children.sort(key=lambda child: child[0])  
 
    if not children:                    
        return maxn(all_pieces, player % 4 + 1, turns + 1, float("inf"), table, tree, parent)[0], None
 
    best_score, best_move, cut = None, None, False
    for distance, old, new, child in children:
        node = None
        if tree is not None and turns < 5: 
            node = str(len(tree))
            tree.create_node(f"P{player}: {coords_in_pos(old)}->{coords_in_pos(new)}", node, parent=parent)
 
        if distance == 0:                        
            scores = score(child)
        else:
            # max score across all 4 players is 20x4 whicch is 80
            if best_score is None:
                child_bound = float("inf")               
            else:
                child_bound = 80 - best_score[player - 1]
            scores = maxn(child, player % 4 + 1, turns + 1, child_bound, table, tree, node)[0]
 
        if node:
            tree[node].tag += f"  {scores}"
        if best_score is None or scores[player - 1] > best_score[player - 1]:
            best_score, best_move = scores, (old, new)
        if best_score[player - 1] >= bound:         
            cut = True
            if node:
                tree[node].tag += "  [pruned rest]"
            break
 
    if not cut:                         
        table[key] = (best_score, best_move)
    return best_score, best_move
 

def AI_Player_Team18(board, player, visualize_tree=0):
    validity_input(board, player)
    all_pieces = tuple(
        tuple(sorted((row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if board[row][col] == p))
        for p in range(1, 5)
    )
 
    tree = None
    if visualize_tree:
        tree = Tree()
        tree.create_node(f"Player {player} to move", "root")
 
    _, move = maxn(all_pieces, player, 0, float("inf"), {}, tree, "root")
    if move is None:
        raise ValueError(f"Player {player} has no legal moves")
 
    if tree is not None:
        save_tree(tree, "Team18_Tree.png")
    return coords_in_pos(move[0]), coords_in_pos(move[1])
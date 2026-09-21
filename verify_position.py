"""
Standalone check: replays the EXACT move log you posted (copy-pasted, not
retyped) up to Player 1's first "E5 -> E3", reconstructs that exact board,
and asks AI_Player_Team24 what it actually does there.

Run:  python3 verify_position.py

Expected (with the fixed file): ('C4', 'C3')
If you see ('E5', 'E3') instead, the fix genuinely isn't active on this
machine -- paste the full output back and we'll dig further.
"""
from halma import initial_pos, move, parse_position
from AI_Player_Team24 import AI_Player_Team24

LOG = """
Player 1: B1 -> B2
Player 2: D1 -> C1
Player 3: E4 -> D4
Player 4: A5 -> C5
Player 1: A2 -> C2
Player 2: E2 -> E3
Player 3: D4 -> E4
Player 4: A4 -> B4
Player 1: B2 -> D2
Player 2: C1 -> D1
Player 3: E4 -> E2
Player 4: B5 -> B3
Player 1: D2 -> D3
Player 2: D1 -> C1
Player 3: D5 -> D4
Player 4: B4 -> B2
Player 1: D3 -> D5
Player 2: C1 -> D1
Player 3: D4 -> D3
Player 4: B3 -> B1
Player 1: A1 -> C1
Player 2: E3 -> E4
Player 3: D3 -> D4
Player 4: B2 -> D2
Player 1: C1 -> C3
Player 2: E1 -> C1
Player 3: E5 -> E3
Player 4: C5 -> E5
Player 1: C2 -> C4
Player 2: C1 -> C2
Player 3: E3 -> D3
Player 4: E5 -> E3
Player 1: C3 -> C5
Player 2: C2 -> C3
Player 3: E2 -> E1
Player 4: D2 -> E2
Player 1: C5 -> E5
Player 2: C3 -> B3
Player 3: E1 -> C1
Player 4: E3 -> E1
Player 1: E5 -> E3
Player 2: B3 -> A3
Player 3: D3 -> C3
Player 4: B1 -> B2
Player 1: E3 -> E5
Player 2: D1 -> D2
Player 3: D4 -> D3
Player 4: B2 -> C2
Player 1: C4 -> D4
Player 2: A3 -> B3
Player 3: C1 -> B1
Player 4: C2 -> C1
"""

board = [row[:] for row in initial_pos]
lines = [l.strip() for l in LOG.strip().splitlines() if l.strip()]

target_index = None
for i, line in enumerate(lines):
    if line.startswith("Player 1: E5 -> E3"):
        target_index = i
        break

if target_index is None:
    raise SystemExit("Could not find 'Player 1: E5 -> E3' in the log -- check the LOG text above.")

for i, line in enumerate(lines):
    header, rest = line.split(": ")
    player = int(header.split()[-1])
    old_ref, new_ref = rest.split(" -> ")
    ok = move(board, parse_position(old_ref), parse_position(new_ref), player)
    if not ok:
        print(f"WARNING: illegal per engine at line {i}: {line}")
    if i == target_index - 1:
        break

print("Reconstructed board just before Player 1's 'E5 -> E3':")
for row in board:
    print(" ", row)
print()

result = AI_Player_Team24([row[:] for row in board], 1, False)
print("AI_Player_Team24 returns:", result)
print("Expected: ('C4', 'C3')")

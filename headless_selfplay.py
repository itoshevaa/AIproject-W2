"""
Quick, pygame-free sanity test. Plays AI_Player_Team24 against random_bot in
all four seats, round-robin, and reports wins/ties/losses/errors and timing.
Run:  python headless_selfplay.py
DO NOT SUBMIT!
"""
import time
from typing import Dict, List, Optional

from halma import check_win_condition, move, random_bot, initial_pos
from AI_Player_Team24 import AI_Player_Team24, pos_to_label
from halma import parse_position

MOVE_LIMIT = 100
GAMES_PER_SEAT = 3


def play_one_game(my_seat: int, verbose: bool = False) -> Dict:
    board = [row[:] for row in initial_pos]
    move_count = 0
    current = 1
    my_move_times: List[float] = []
    errors = 0

    while True:
        result = check_win_condition(board, move_count, MOVE_LIMIT, True)
        if result.status != "ongoing":
            return {
                "status": result.status,
                "winners": result.winners,
                "losers": result.losers,
                "tied": result.tied_players,
                "move_count": move_count,
                "my_move_times": my_move_times,
                "errors": errors,
            }

        bot_fn = AI_Player_Team24 if current == my_seat else random_bot
        try:
            old_ref, new_ref = bot_fn([row[:] for row in board], current, False)
            if current == my_seat:
                pass  # timing measured below
            old_pos = parse_position(old_ref)
            new_pos = parse_position(new_ref)
            ok = move(board, old_pos, new_pos, current)
            if not ok:
                errors += 1 if current == my_seat else 0
        except Exception as e:
            if current == my_seat:
                errors += 1
            if verbose:
                print(f"Player {current} errored: {e}")

        move_count += 1
        current = current % 4 + 1


def timed_move(board, player):
    start = time.perf_counter()
    result = AI_Player_Team24([row[:] for row in board], player, False)
    return result, time.perf_counter() - start


def main() -> None:
    print("=== Timing a handful of individual moves from the start position ===")
    board = [row[:] for row in initial_pos]
    for _ in range(3):
        (old_ref, new_ref), elapsed = timed_move(board, 1)
        print(f"Move: {old_ref} -> {new_ref}   ({elapsed:.3f}s)")
        old_pos, new_pos = parse_position(old_ref), parse_position(new_ref)
        move(board, old_pos, new_pos, 1)

    print("\n=== Self-play: AI_Player_Team24 vs three random bots, all seats ===")
    total_errors = 0
    for seat in (1, 2, 3, 4):
        wins = ties = losses = 0
        for g in range(GAMES_PER_SEAT):
            outcome = play_one_game(seat)
            total_errors += outcome["errors"]
            if seat in outcome["winners"]:
                wins += 1
            elif seat in outcome["tied"]:
                ties += 1
            else:
                losses += 1
            print(f"  seat {seat} game {g+1}: status={outcome['status']} "
                  f"winners={outcome['winners']} moves={outcome['move_count']} "
                  f"errors_this_game={outcome['errors']}")
        print(f"Seat {seat} summary: {wins} wins, {ties} ties, {losses} losses/other "
              f"out of {GAMES_PER_SEAT} games")

    print(f"\nTotal AI errors across all games: {total_errors}")


if __name__ == "__main__":
    main()

"""
In order to run this file:

Install pygame using pip 
Run:     python halma_pygame.py
"""
from typing import List, Optional, Tuple, Callable
import pygame

from halma import (
    check_legal_move,
    check_win_condition,
    move,
    parse_position,
    random_bot,
    illegal_bot,
    win_cells_1v1,
    initial_pos_1v1
)

# True: human player 1 versus bot player 2. False: two local humans.
PLAY_AGAINST_BOT: bool = True
MAXIMUM_MOVE_LIMIT: int = 10  # Total successful moves, across both players.
VISUALIZE_SEARCH_TREE: bool = False # This is here to match the function signature, the random bot does not visualize anything 
BOT_DELAY_MS: int = 450

INITIAL_BOARD: List[List[int]] = initial_pos_1v1

CELL_SIZE = 90
BOARD_X = 55
BOARD_Y = 135
WINDOW_SIZE = (560, 710)
COLORS = {1: (58, 124, 223), 2: (229, 115, 55)}
ZONE_COLORS = {1: (204, 222, 249), 2: (250, 220, 198)}

BotFunction = Callable[
    [List[List[int]], int, bool],
    Tuple[str, str]
]

# You can set your bot function here, the default one is random_bot, which as the name suggests makes random moves
BOT_FUNCTION: BotFunction = illegal_bot


class HalmaGame:
    def __init__(self, play_against_bot: bool = PLAY_AGAINST_BOT) -> None:
        self.play_against_bot = play_against_bot
        self.reset()

    def reset(self) -> None:
        self.board = [row[:] for row in INITIAL_BOARD]
        self.current_player = 1
        self.move_count = 0
        self.selected: Optional[Tuple[int, int]] = None
        self.message = "Select your piece, then a highlighted square."
        self.bot_error = False
        self.result = check_win_condition(
            self.board, self.move_count, MAXIMUM_MOVE_LIMIT, False
        )
        self.bot_due = pygame.time.get_ticks() + BOT_DELAY_MS

    def destinations(self, old_position: Tuple[int, int]) -> List[Tuple[int, int]]:
        return [
            (row, column)
            for row in range(5)
            for column in range(5)
            if check_legal_move(self.board, old_position, (row, column))
        ]

    def attempt_move(
        self, old_position: Tuple[int, int], new_position: Tuple[int, int]
    ) -> bool:
        if self.result.status != "ongoing" or self.bot_error:
            return False
        if not move(self.board, old_position, new_position, self.current_player):
            self.message = "Illegal move. Choose a highlighted square."
            return False

        self.move_count += 1
        self.selected = None
        self.result = check_win_condition(
            self.board, self.move_count, MAXIMUM_MOVE_LIMIT, False
        )
        if self.result.status == "ongoing":
            self.current_player = 3 - self.current_player
        self.message = "Select your piece, then a highlighted square."
        self.bot_due = pygame.time.get_ticks() + BOT_DELAY_MS
        return True

    def handle_click(self, mouse_position: Tuple[int, int]) -> None:
        if self.result.status != "ongoing" or self.bot_error:
            return
        if self.play_against_bot and self.current_player == 2:
            return

        x, y = mouse_position
        if not (BOARD_X <= x < BOARD_X + 5 * CELL_SIZE
                and BOARD_Y <= y < BOARD_Y + 5 * CELL_SIZE):
            return
        position = ((y - BOARD_Y) // CELL_SIZE, (x - BOARD_X) // CELL_SIZE)
        row, column = position
        if self.board[row][column] == self.current_player:
            self.selected = None if position == self.selected else position
            self.message = "Select your piece, then a highlighted square."
        elif self.selected is not None:
            self.attempt_move(self.selected, position)

    def update_bot(self) -> None:
        if (not self.play_against_bot or self.current_player != 2
                or self.result.status != "ongoing" or self.bot_error
                or pygame.time.get_ticks() < self.bot_due):
            return
        try:
            
            # Give the bot a copy so its search cannot change the live board. This is where the bot function you provided will be executed. DO NOT CHANGE IT HERE, DO IT ABOVE
            old_reference, new_reference = BOT_FUNCTION(
                [row[:] for row in self.board],
                self.current_player,
                VISUALIZE_SEARCH_TREE,
            )
            print(f"OLD: {old_reference}")
            print(f"NEW: {new_reference}")
            
            old_position = parse_position(old_reference)
            new_position = parse_position(new_reference)
            if not self.attempt_move(old_position, new_position):
                raise ValueError("Bot returned an illegal move")
        except (ValueError, TypeError, IndexError) as error:
            print(f"Bot error: {error}")
            self.bot_error = True
            self.message = "Bot error: see console. Press R to restart."

    def draw(self, screen: pygame.Surface, font: pygame.font.Font,
             small_font: pygame.font.Font) -> None:
        screen.fill((245, 247, 251))

        def label(text: str, x: int, y: int, small: bool = False) -> None:
            screen.blit((small_font if small else font).render(
                text, True, (30, 39, 55)), (x, y))

        label("HALMA | 1 vs 1", 35, 20)
        mode = "Human vs random bot" if self.play_against_bot else "Human vs human"
        label(f"{mode}   |   Moves: {self.move_count}/{MAXIMUM_MOVE_LIMIT}",
              35, 58, True)

        if self.result.status == "winner":
            status = "Winner: Player " + ", ".join(map(str, self.result.winners))
        elif self.result.status == "move_limit":
            status = "Game ended: move limit reached"
        elif self.bot_error:
            status = "Game paused: bot error"
        else:
            actor = " (bot)" if self.play_against_bot and self.current_player == 2 else ""
            status = f"Player {self.current_player}'s turn{actor}"
        label(status, 35, 84)

        legal = self.destinations(self.selected) if self.selected is not None else []
        for index in range(5):
            label(chr(ord("A") + index), BOARD_X + index * CELL_SIZE + 36,
                  BOARD_Y - 27, True)
            label(str(index + 1), BOARD_X - 25,
                  BOARD_Y + index * CELL_SIZE + 34, True)

        for row in range(5):
            for column in range(5):
                position = (row, column)
                rect = pygame.Rect(BOARD_X + column * CELL_SIZE,
                                   BOARD_Y + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                background = (255, 255, 255) if (row + column) % 2 == 0 else (234, 239, 246)
                for player, cells in win_cells_1v1.items():
                    if position in cells:
                        background = ZONE_COLORS[player]
                pygame.draw.rect(screen, background, rect)
                pygame.draw.rect(screen, (179, 190, 205), rect, 1)
                if position == self.selected:
                    pygame.draw.rect(screen, (241, 185, 42), rect.inflate(-6, -6), 4)
                if position in legal:
                    pygame.draw.circle(screen, (65, 156, 110), rect.center, 10)
                piece = self.board[row][column]
                if piece:
                    pygame.draw.circle(screen, COLORS[piece], rect.center, 28)
                    number = font.render(str(piece), True, (255, 255, 255))
                    screen.blit(number, number.get_rect(center=rect.center))

        if self.result.status == "move_limit":
            tied = ", ".join(map(str, self.result.tied_players)) or "none"
            lost = ", ".join(map(str, self.result.losers)) or "none"
            label(f"Tied players: {tied}", 35, 602, True)
            label(f"Lost for blocking: {lost}", 35, 628, True)
        else:
            label(self.message, 35, 602, True)
            label("Tinted squares are the matching player's end zone.", 35, 628, True)
        label("R: restart    |    Esc: quit", 35, 670, True)


def main(play_against_bot: bool = PLAY_AGAINST_BOT) -> None:
    pygame.init()
    try:
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Halma - 1 vs 1")
        font = pygame.font.Font(None, 30)
        small_font = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        game = HalmaGame(play_against_bot)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    game.handle_click(event.pos)
            if not running:
                break
            game.update_bot()
            game.draw(screen, font, small_font)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

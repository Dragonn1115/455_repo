from board_base import EMPTY, GO_COLOR, GO_POINT
from board import GoBoard
from engine import GoEngine

class Go3Args:
    def __init__(self, sim: int, move_select: str, sim_rule: str, 
                 check_selfatari: bool, limit: int) -> None:
        self.sim: int = sim
        self.move_select: str = move_select
        self.use_ucb: bool = (move_select != "simple")
        self.sim_rule: str = sim_rule
        self.random_simulation: bool = (sim_rule == "random")
        self.check_selfatari: bool = check_selfatari
        self.use_pattern: bool = not self.random_simulation
        self.limit: int = limit

class GoSimulationEngine(GoEngine):
    def __init__(self, name: str, version: float, 
                 sim: int, move_select: str, sim_rule: str, 
                 check_selfatari: bool, limit: int = 100) -> None:
        """
        Go player that selects moves by simulation.
        """
        GoEngine.__init__(self, name, version)
        self.args: Go3Args = Go3Args(sim, move_select, sim_rule, 
                          check_selfatari, limit)

    def simulate(self, board: GoBoard, move: GO_POINT, toplay: GO_COLOR) -> GO_COLOR:
        """
        Run a simulated game for a given move.
        """
        cboard: GoBoard = board.copy()
        cboard.play_move(move, toplay)
        opp: GO_COLOR = opponent(toplay)
        return self.playGame(cboard, opp)
        
    def simulateMove(self, board: GoBoard, move: GO_POINT, toplay: GO_COLOR) -> int:
        """
        Run self.sim simulations for a given move. Returns number of wins.
        """
        wins = 0
        for _ in range(self.args.sim):
            result = self.simulate(board, move, toplay)
            if result == toplay:
                wins += 1
        return wins

    def playGame(self, board: GoBoard, color: GO_COLOR) -> GO_COLOR:
        """
        Run a simulation game.
        """
        nuPasses = 0
        for _ in range(self.args.limit):
            color = board.current_player
            if self.args.random_simulation:
                move = GoBoardUtil.generate_random_move(board, color, True)
            else:
                move = PatternUtil.generate_move_with_filter(
                    board, self.args.use_pattern, self.args.check_selfatari
                )
            board.play_move(move, color)
            if move == PASS:
                nuPasses += 1
            else:
                nuPasses = 0
            if nuPasses >= 2:
                break
        return winner(board, self.komi)
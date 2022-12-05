#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

from gtp_connection import GtpConnection
from board_util import GoBoardUtil
from board import GoBoard


from board_util import (
    GoBoardUtil,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    PASS,
    is_black_white,
    is_black_white_empty,
    coord_to_point,
    where1d,
    MAXSIZE,
    GO_POINT
)

from math import log, sqrt
from typing import List, Tuple
STATS = List[List[int]]
INFINITY = float("inf")

def mean(stats: STATS, i: int) -> float:
    return stats[i][0] / stats[i][1]

def ucb(stats: STATS, C: float, i: int, n: int) -> float:
    if stats[i][1] == 0:
        return INFINITY
    return mean(stats, i) + C * sqrt(log(n) / stats[i][1])

def findBest(stats: STATS, C: float, n: int) -> int:
    best = -1
    bestScore = -INFINITY
    for i in range(len(stats)):
        score = ucb(stats, C, i, n)
        if score > bestScore:
            bestScore = score
            best = i
    assert best != -1
    return best


class Go0:
    def __init__(self):
        """
        NoGo player that selects moves randomly from the set of legal moves.

        Parameters
        ----------
        name : str
            name of the player (used by the GTP interface).
        version : float
            version number (used by the GTP interface).
        """
        self.name = "Go0"
        self.version = 1.0
        self.selection = "rr"
        self.sim = 10

    def get_move(self, state, color):

        assert not state.endOfGame()
        moves = GoBoardUtil.generate_legal_moves(state, color)
        numMoves = len(moves)
        if self.selection == "rr":
            self.sim = 10
        elif self.selection == "ucb":
            self.sim = 10 * numMoves
            # C = 0.4  # sqrt(2) is safe, this is more aggressive
            # best = runUcb(self, state, C, moves, color)
            # return best

        score = [0] * numMoves
        stats = [[0, 0] for _ in moves]

        for i in range(numMoves):
            if self.selection == 'ucb':
                C = 0.4
                moveIndex = findBest(stats, C, i)
                move = moves[moveIndex]
            else:
                move = moves[i]
            score[i] = self.simulate(state, move)

        bestIndex = score.index(max(score))
        best = moves[bestIndex]
        assert best in state.legalMoves()

        return best

    def simulate(self, state, move):
        stats = [0] * 3
        state.play_move(move,state.current_player)
        moveNr = state.moveNumber()
        for _ in range(self.sim):
            winner, _ = state.simulate()
            stats[winner] += 1
            state.resetToMoveNumber(moveNr)
        assert sum(stats) == self.sim
        assert moveNr == state.moveNumber()
        state.undoMove()
        eval = (stats[BLACK] + 0.5 * stats[EMPTY]) / self.sim
        if state.current_player == WHITE:
            eval = 1 - eval
        return eval





def run():
    """
    start the gtp connection and wait for commands.
    """
    board = GoBoard(7)
    con = GtpConnection(Go0(), board)
    con.start_connection()

if __name__ == "__main__":
    run()

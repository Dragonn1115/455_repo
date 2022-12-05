"""
feature_moves.py
Move generation based on simple features.
"""
from board_base import GO_COLOR, GO_POINT, NO_POINT
from board import GoBoard
from board_score import winner
from board_util import GoBoardUtil, PASS

import numpy as np
import random
from typing import Any, Tuple, List

MOVES_PROBS_LIST = Tuple[List[GO_POINT], np.ndarray]

class FeatureMoves(object):

    @staticmethod
    def playGame(board: GoBoard, color: GO_COLOR, **kwargs: Any) -> GO_COLOR:
    # TODO is Any OK here? See discussion in
    # https://stackoverflow.com/questions/37031928/type-annotations-for-args-and-kwargs
        """
        Run a simulation game according to given parameters.
        """
        # komi = kwargs.pop("komi", 0)
        limit = kwargs.pop("limit", 1000)
        # simulation_policy = kwargs.pop("random_simulation", "random")
        # use_pattern = kwargs.pop("use_pattern", True)
        # check_selfatari = kwargs.pop("check_selfatari", True)
        # if kwargs:
        #     raise TypeError("Unexpected **kwargs: %r" % kwargs)

        for _ in range(limit):
            color = board.current_player

            move = board.generate_simulate_move()
            if move == None:
                winner = board.true_winner()
                return winner
            board.play_move(move, color)

        winner = board.score_winner()
        return winner
        # return winner, board.weight

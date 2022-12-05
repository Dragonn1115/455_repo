"""
board.py

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""
import random
import numpy as np
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

"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See GoBoardUtil.coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size):
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)

    def reset(self, size):
        """
        Creates a start state, an empty board with given size.
        """
        self.size = size
        self.NS = size + 1
        self.WE = 1
        self.last_move = None
        self.last2_move = None
        self.current_player = BLACK
        self.maxpoint = size * size + 3 * (size + 1)
        self.board = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.policy = "random"
        self.board_moves = []

    def copy(self):
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point):
        return self.board[point]

    def pt(self, row, col):
        return coord_to_point(row, col, self.size)

    def is_legal(self, point, color):
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        board_copy = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def get_empty_points(self):
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row):
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board):
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start = self.row_start(row)
            board[start : start + self.size] = EMPTY

    def is_eye(self, point, color):
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = GoBoardUtil.opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point, color):
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block):
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone):
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point):
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=bool)
        pointstack = [point]
        color = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point):
        """
        Check whether opponent block on nb_point is captured.
        Return a boolean
        True: The block is captured
        False: The block is not captured 
        """
        opp_block = self._block_of(nb_point)
        return not self._has_liberty(opp_block)

    def play_move(self, point, color):
        """
        Play a move of color on point
        Returns boolean: whether move was legal
        """
        assert is_black_white(color)
        
        # Special cases
        if point == PASS:
            return False
        elif self.board[point] != EMPTY:
            return False

        opp_color = GoBoardUtil.opponent(color)
        self.board[point] = color
        neighbors = self._neighbors(point)
        # check for capturing
        for nb in neighbors:
            if self.board[nb] == opp_color:
                captured = self._detect_and_process_capture(nb)
                if captured:
                    # undo capturing move
                    self.board[point] = EMPTY
                    return False
        
        # check for suicide
        block = self._block_of(point)
        if not self._has_liberty(block):  # undo suicide move
            self.board[point] = EMPTY
            return False

        self.current_player = GoBoardUtil.opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        self.board_moves.append(point)

        return True

    def winner(self):
        moves = self.legalMoves()
        if len(moves) == 0:
            return GoBoardUtil.opponent(self.current_player)
        else:
            return None

    def undoMove(self):
        move = self.board_moves.pop()
        self.board[move] = EMPTY
        self.current_player = GoBoardUtil.opponent(self.current_player)

    def neighbors_of_color(self, point, color):
        """ List of neighbors of point of given color """
        nbc = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point):
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point):
        """ List of all four diagonal neighbors of point """
        return [
            point - self.NS - 1,
            point - self.NS + 1,
            point + self.NS - 1,
            point + self.NS + 1,
        ]

    def last_board_moves(self):
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not None, not PASS).
        """
        board_moves = []
        if self.last_move != None and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != None and self.last2_move != PASS:
            board_moves.append(self.last2_move)
            return 

    def legalMoves(self):
        return GoBoardUtil.generate_legal_moves(self, self.current_player)

    def generate_neighbors_list(self, pt):
        diag = self._diag_neighbors(pt)
        hori = self._neighbors(pt)
        neighbors = [diag[0],hori[2],diag[1],hori[0],hori[1],diag[2],hori[3],diag[3]]
        colors = []
        for point in neighbors:
            colors.append(self.board[point])
        return colors

    def endOfGame(self):
        moves = self.legalMoves()
        if len(moves) == 0:
            return True    
        return False

    # simulate one game from the current state until the end
    def simulate(self):
        i = 0
        if not self.endOfGame():
            while not self.endOfGame():
                allMoves = self.legalMoves()
                if self.policy == "random":
                    random.shuffle(allMoves)
                    self.play_move(allMoves[0],self.current_player)

                elif self.policy == "pattern":
                    prob = []
                    for k in range(len(allMoves)):

                        code_list = self.generate_neighbors_list(allMoves[k])
                        reversed(code_list)
                        x = 0
                        for j in range(len(code_list)):
                            x += code_list[j]*(pow(4,j))
                        prob.append(float(self.weights[x]))
                        final = [round(prob[j]/sum(prob),3) for j in range(len(prob))]
                        
                    X = random.randint(0,1)
                    Y = 0
                    for j in range(len(final)):
                        Y+= final[j]
                        if Y > X:
                            self.play_move(allMoves[j],self.current_player)
                            break

                i += 1
        return self.winner(), i

    def resetToMoveNumber(self, moveNr):
        numUndos = self.moveNumber() - moveNr
        assert numUndos >= 0
        for _ in range(numUndos):
            self.undoMove()
        assert self.moveNumber() == moveNr

    def moveNumber(self):
        return len(self.board_moves)

#!/usr/bin/env python3
"""
Quoridor agent.
Copyright (C) 2013, Marco Novaes 2166579 & Mathurin Chritin 1883619

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""
import math
from time import time

import utils
from quoridor import *


class MyAgent(Agent):
    """
    Quoridor agent.
    This agent implements a minimax tree search, coupled with an alpha-beta
    pruning mechanism and a B-type strategy.
    """

    def __init__(self):
        """
        Initialisation of the agent. Mostly sets the agent parameters.
        """
        # ========= MAIN VARIABLES ===========
        self.main_cut_off_depth = 6
        self.wall_search_zone_size = 3
        self.steps_before_exploring_vw = 8
        self.time_threshold_1_config = (5, 3)  # (threshold [s], new_cutoffdepth)
        self.time_threshold_2_config = (40, 4)  # (threshold [s], new_cutoffdepth)
        self.time_threshold_3_config = (60, 5)  # (threshold [s], new_cutoffdepth)

        self.max_search_duration = 60  # s
        self.play_simple_until_step = 10
        self.max_walls_diff_allowed = 6
        # ====================================

        # ===================== STRATEGIES VARIABLES =====================
        self.use_bfs_aStar = True
        self.use_strategy_typeB = True
        self.nb_actions_per_node = 4 if self.use_strategy_typeB else None
        # ================================================================

        # ==== CONSTANTS ====
        self.big_heuristic_score = 100
        self.my_player = None
        # ===================

        # ============= STATE VARIABLES ==============
        self.cut_off_depth = self.main_cut_off_depth
        self.starting_time = 0
        self.expanded_nodes = 0
        # ============================================

    def has_no_time_left(self):
        """
        Determines if the agent surpassed self.max_search_duration.
        :return:: True if it has, False if it has not
        """
        has_no_time_left = (time() - self.starting_time) >= self.max_search_duration
        if has_no_time_left:
            print(f"Max search time consumed ({self.max_search_duration})")
        return has_no_time_left

    def possible_wall_moves(self, state, current_player, step):
        """
        Returns interesting wall moves from a state. It will return all the wall moves
        in a certain area defined around the two players and self.wall_search_zone_size
        :param state: the current state
        :param current_player: the current player
        :param step: the current step
        :return: moves: a list of the wall moves defined as above
        """
        moves = set()
        if state.nb_walls[current_player] <= 0:
            return moves

        def add_wall_at_pos(position, moves_set):
            if state.is_wall_possible_here(position, True):
                moves_set.add(('WH', position[0], position[1]))
            if step > self.steps_before_exploring_vw and state.is_wall_possible_here(position, False):
                moves_set.add(('WV', position[0], position[1]))

        adv = (current_player + 1) % 2
        my_position = state.pawns[current_player]
        adv_position = state.pawns[adv]

        wall_coords = set()
        grid_size = state.size

        def generate_wall_coords(position: tuple):
            return range(max(position[0] - self.wall_search_zone_size, 0),
                         min(position[0] + self.wall_search_zone_size, grid_size - 1)), \
                   range(max(position[1] - self.wall_search_zone_size, 0),
                         min(position[1] + self.wall_search_zone_size, grid_size - 1))

        rx, ry = generate_wall_coords(adv_position)
        [wall_coords.add((i, j)) for i, j in zip(rx, ry)]

        rx, ry = generate_wall_coords(my_position)
        [wall_coords.add((i, j)) for i, j in zip(rx, ry)]

        walls_already_here = set(state.horiz_walls).union(state.verti_walls)
        [add_wall_at_pos((i, j), moves) for (i, j) in wall_coords if (i, j) not in walls_already_here]

        return moves

    def select_possible_actions(self, state, current_player, step):
        """
        Returns all the interesting moves from a state : all the possible pawn moves
        plus the interesting wall moves as described above. It avoids evaluating
        wall moves if the adversary has more than self.max_walls_diff_allowed than us.
        :param state: the current state
        :param current_player: the current player
        :param step: the current step
        :return: moves: a list containing all the interesting moves from state
        """
        moves = []
        moves.extend(state.get_legal_pawn_moves(current_player))
        if current_player == self.my_player and \
                state.nb_walls[(current_player + 1) % 2] - state.nb_walls[
            current_player] <= self.max_walls_diff_allowed:
            moves.extend(self.possible_wall_moves(state, current_player, step))

        return moves

    def init_config(self, step, time_left, pawn_positions, player, walls_on_map):
        """
        Initialises various parameters from the time remaining, the current step, and other
        characteristics of the current state, in order to have a good time and performance
        compromise.
        :param step: the current step
        :param time_left: time credit remaining for the player
        :param pawn_positions: pawn positions on the board
        :param player: current player
        :param walls_on_map: current walls on the board
        :return: None
        """
        self.cut_off_depth = self.main_cut_off_depth
        self.starting_time = time()
        self.expanded_nodes = 0

        crossed = pawn_positions[player][0] < pawn_positions[(player + 1) % 2][0]
        crossed = not crossed if player == 0 else crossed

        if step < self.play_simple_until_step and not crossed and walls_on_map == 0:
            self.cut_off_depth = 4
            print(f"Reducing cutoff to {self.cut_off_depth}")
        else:
            self.cut_off_depth = self.main_cut_off_depth

        if time_left < self.time_threshold_1_config[0]:
            self.cut_off_depth = self.time_threshold_1_config[1]
            print(f"SETTING CUTOFF DEPTH TO {self.cut_off_depth} TO BE QUICKER")
        elif time_left < self.time_threshold_2_config[0]:
            self.cut_off_depth = self.time_threshold_2_config[1]
            print(f"SETTING CUTOFF DEPTH TO {self.cut_off_depth} TO BE QUICKER")
        elif time_left < self.time_threshold_3_config[0]:
            self.cut_off_depth = self.time_threshold_3_config[1]
            print(f"SETTING CUTOFF DEPTH TO {self.cut_off_depth} TO BE QUICKER")

    def critical_situation(self, state, step):
        """
        Check if some player can win in his next move
        If I can win, we choose the actions that makes me win
        If adversary can win, we choose if possible the action that blocks the adversary's victory
        :param state: the current state
        :param step: the current step
        :return: True and a critical move to play now or False and None
        """

        def get_victory_move(player):
            pawn_actions = state.get_legal_pawn_moves(player)
            for action in pawn_actions:
                if action[1] == state.goals[player]:
                    return True, action
            return False, None

        if step < 14:
            return False, None

        if abs(state.pawns[self.my_player][0] - state.goals[self.my_player]) <= 2:
            can_i_win, victory_move = get_victory_move(self.my_player)
            if can_i_win:
                return True, victory_move

        adv = int(not self.my_player)
        if not abs(state.pawns[adv][0] - state.goals[adv]) <= 2:
            return False, None

        can_adv_win, victory_move = get_victory_move(adv)
        if not (can_adv_win and state.nb_walls[self.my_player] > 0):
            return False, None

        # 1º kind of victory move: one step forward
        if abs(state.pawns[adv][0] - state.goals[adv]) == 1 and state.pawns[adv][1] == victory_move[2]:
            positions = [[min(state.pawns[adv][0], victory_move[1]), victory_move[2] - 1],
                         [min(state.pawns[adv][0], victory_move[1]), victory_move[2]]]
            for position in positions:
                if state.is_wall_possible_here(position, True):
                    return True, ('WH', position[0], position[1])

        # 2º kind of victory move: one diagonal step forward
        elif abs(state.pawns[adv][1] - victory_move[2]) == 1:
            position = [min(state.pawns[adv][0], victory_move[1]), min(state.pawns[adv][1], victory_move[2])]
            if state.is_wall_possible_here(position, True):
                return True, ('WH', position[0], position[1])
            elif state.is_wall_possible_here(position, False):
                return True, ('WV', position[0], position[1])

        # 3º kind of victory move: two steps forward
        elif abs(state.pawns[adv][0] - state.goals[adv]) == 2:
            if self.my_player == 0:
                positions = [[0, state.pawns[adv][1] - 1], [0, state.pawns[adv][1]],
                             [1, state.pawns[adv][1] - 1], [1, state.pawns[adv][1]]]
            else:
                positions = [[7, state.pawns[adv][1] - 1], [7, state.pawns[adv][1]],
                             [6, state.pawns[adv][1] - 1], [6, state.pawns[adv][1]]]
            for position in positions:
                if state.is_wall_possible_here(position, True):
                    return True, ('WH', position[0], position[1])

        return False, None

    def heuristic(self, state, depth):
        """
        Evaluates the interesting-ness of a state.
        :param state: the state to evaluate
        :param depth: the current depth in the search tree
        :return: int: a score representing the evaluation of state
        """
        if state.is_finished() and depth == 0:
            print("Forcing win because we can win")
            return self.big_heuristic_score

        me = self.my_player
        adv = (me + 1) % 2

        if self.use_bfs_aStar:
            try:
                min_steps_before_victory_adv = utils.min_steps_before_victory_aStar_simplified(state, adv)
            except NoPath:
                print("No Path for adv - A*")
                return 0
            try:
                min_steps_before_victory_me = utils.min_steps_before_victory_aStar_simplified(state, me)
            except NoPath:
                print("No Path for me - A*")
                return 0
        else:
            try:
                min_steps_before_victory_adv = state.min_steps_before_victory(adv)
            except NoPath:
                print("No Path for adv")
                return 0
            try:
                min_steps_before_victory_me = state.min_steps_before_victory(me)
            except NoPath:
                print("No Path for me")
                return 0

        walls_diff = state.nb_walls[adv] - state.nb_walls[me]
        distance_diff = min_steps_before_victory_adv - min_steps_before_victory_me
        final_val = distance_diff if distance_diff != 0 else -0.5 * walls_diff

        return final_val

    def min_value(self, state, alpha, beta, current_player, depth, step):
        """
        Minimizes the score of the subtree.
        :param state: current subtree root
        :param alpha: alpha threshold
        :param beta: beta threshold
        :param current_player: the current player
        :param depth: the current depth
        :param step: the current step
        :return: best_score, best_move: the best move found in the subtree with its associated score.
        """
        if self.has_no_time_left():
            print("Setting cut_off_depth to 2")
            self.cut_off_depth = 2

        if depth >= self.cut_off_depth or state.is_finished():
            return self.heuristic(state, depth), ('P', -1, -1)
        best_score = math.inf
        best_move = ('P', -1, -1)
        actions = self.select_possible_actions(state, current_player, step)

        if self.use_strategy_typeB:
            states = [state.clone().play_action(action, current_player) for action in actions]
            scores = [self.heuristic(state, depth) for state in states]
            action_state = [[actions[el], states[el]] for el in range(len(scores))]
            nb_states = min(self.nb_actions_per_node, len(actions))
            selected_action_states = [action_state[idx] for idx in
                                      sorted(range(len(scores)), key=lambda x: scores[x])[:nb_states]]

        else:
            selected_action_states = [[action, state.clone().play_action(action, current_player)] for action in
                                      actions]

        for action, newState in selected_action_states:
            self.expanded_nodes += 1
            val, _ = self.max_value(newState, alpha, beta, int(not current_player), depth + 1, step)
            if val < best_score:
                best_score = val
                best_move = action
                beta = min(best_score, beta)
            if best_score <= alpha:
                return best_score, best_move

        return best_score, best_move

    def max_value(self, state, alpha, beta, current_player, depth, step):
        """
        Maximizes the score of the subtree.
        :param state: current subtree root
        :param alpha: alpha threshold
        :param beta: beta threshold
        :param current_player: the current player
        :param depth: the current depth
        :param step: the current step
        :return: best_score, best_move: the best move found in the subtree with its associated score.
        """
        if self.has_no_time_left():
            print("Setting cut_off_depth to 2")
            self.cut_off_depth = 2

        if depth >= self.cut_off_depth or state.is_finished():
            return self.heuristic(state, depth), ('P', -1, -1)
        best_score = -math.inf
        best_move = ('P', -1, -1)
        actions = self.select_possible_actions(state, current_player, step)

        if self.use_strategy_typeB:
            states = [state.clone().play_action(action, current_player) for action in actions]
            scores = [self.heuristic(state, depth) for state in states]
            action_state = [[actions[el], states[el]] for el in range(len(scores))]
            nb_states = min(self.nb_actions_per_node, len(actions))
            selected_action_states = [action_state[idx] for idx in
                                      sorted(range(len(scores)), key=lambda x: scores[x])[-nb_states:]]
            selected_action_states.reverse()

        else:
            selected_action_states = [[action, state.clone().play_action(action, current_player)] for action in
                                      actions]

        for action, newState in selected_action_states:
            self.expanded_nodes += 1
            val, _ = self.min_value(newState, alpha, beta, int(not current_player), depth + 1, step)
            if val > best_score:
                best_score = val
                best_move = action
                alpha = max(alpha, best_score)
            if best_score >= beta:
                return best_score, best_move
        return best_score, best_move

    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in quoridor.py.
        :param player: the player to control in this step (0 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
          eg: ('P', 5, 2) to move your pawn to cell (5,2)
          eg: ('WH', 5, 2) to put a horizontal wall on corridor (5,2)
          for more details, see `Board.get_actions()` in quoridor.py
        """
        print("-------- START STEP --------")

        def alpha_beta_minimax_search():
            return self.max_value(board, -math.inf, math.inf, player, 0, step)

        self.my_player = player
        board = dict_to_board(percepts)
        self.init_config(step, time_left, board.pawns, player, len(board.verti_walls) + len(board.horiz_walls))
        board.pretty_print()
        print("player:", player, "(red)" if player else "(blue)")
        print("step:", step)
        print("time left:", time_left if time_left else '+inf')
        print("Cut_off_depth is set to:", self.cut_off_depth)

        # do the right action if some player can win with his next move
        can_player_win, critical_move = self.critical_situation(board, step)
        if can_player_win:
            return critical_move

        # when adversary starts, save time on the 3 first moves by advancing while the situation is « safe »
        if (step == 2 or step == 4 or step == 6) and len(board.verti_walls) == 0 and len(board.horiz_walls) == 0:
            current_position = board.pawns[player]
            move = ('P', current_position[0] + (1 if player == 0 else -1), current_position[1])
            score = -99
        # else, run the main research !
        else:
            score, move = alpha_beta_minimax_search()

        print("Returning move", move)
        print("Move score", score)
        print("Total elapsed time: {:.6f}".format(time() - self.starting_time))
        print("Nodes expanded (TOTAL)", self.expanded_nodes)
        print("--------- END STEP ---------\n")
        print('', flush=True)
        return move


if __name__ == "__main__":
    agent_main(MyAgent())

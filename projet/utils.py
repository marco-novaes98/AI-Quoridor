"""
@authors Marco Novaes 2166579 & Mathurin Chritin 1883619
Utility functions. We use them in our agent to accelerate the evaluation
of the shortest paths on the board. Some of the functions/classes were
taken from our older projects.
"""

import heapq


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.

      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """

    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def exists(self, item):
        return item in (x[1][0] for x in self.heap)


class NoPath(Exception):
    """Raised when a player puts a wall such that no path exists
    between a player and its goal row"""

    def __repr__(self):
        return "Exception: no path to reach the goal"


def get_shortest_path_aStar(state, player):
    """ Returns a shortest path for player to reach its goal
    if player is on its goal, the shortest path is an empty list
    if no path exists, exception is thrown.
    """

    def get_pawn_moves(pos):
        (x, y) = pos
        positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                     (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1),
                     (x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)]
        moves = []
        for new_pos in positions:
            if state.is_pawn_move_ok(pos, new_pos,
                                     state.pawns[(player + 1) % 2]):
                moves.append(new_pos)
        return moves

    (a, b) = state.pawns[player]
    if a == state.goals[player]:
        return []
    visited = [[False for _ in range(state.size)] for _ in range(state.size)]
    # Predecessor matrix in the BFS
    prede = [[None for _ in range(state.size)] for _ in range(state.size)]
    neighbors = PriorityQueue()
    neighbors.push((state.pawns[player], 0), a)  # (state, dept), cost)
    while not neighbors.isEmpty():
        neighbor, depth = neighbors.pop()
        (x, y) = neighbor
        visited[x][y] = True
        if x == state.goals[player]:
            succ = [neighbor]
            curr = prede[x][y]
            while curr is not None and curr != state.pawns[player]:
                succ.append(curr)
                (x_, y_) = curr
                curr = prede[x_][y_]
            succ.reverse()
            return succ
        unvisited_succ = [(x_, y_) for (x_, y_) in
                          get_pawn_moves(neighbor) if not visited[x_][y_]]
        for n_ in unvisited_succ:
            (x_, y_) = n_
            if not neighbors.exists(n_):
                neighbors.push((n_, depth + 1), abs(n_[0] - state.goals[player]) + depth + 1)
                prede[x_][y_] = neighbor

    raise NoPath()


def get_simplified_shortest_path_aStar(state, player):
    """ Returns a shortest path for player to reach its goal
    if player is on its goal, the shortest path is an empty list
    if no path exists, exception is thrown.
    """

    def get_pawn_moves(pos):
        (x, y) = pos
        positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        moves = []
        for new_pos in positions:
            if state.is_simplified_pawn_move_ok(pos, new_pos):
                moves.append(new_pos)
        return moves

    (a, b) = state.pawns[player]
    if a == state.goals[player]:
        return []
    visited = [[False for _ in range(state.size)] for _ in range(state.size)]
    # Predecessor matrix in the BFS
    prede = [[None for _ in range(state.size)] for _ in range(state.size)]
    neighbors = PriorityQueue()
    neighbors.push((state.pawns[player], 0), a)  # (state, dept), cost)
    while not neighbors.isEmpty():
        neighbor, depth = neighbors.pop()
        (x, y) = neighbor
        visited[x][y] = True
        if x == state.goals[player]:
            succ = [neighbor]
            curr = prede[x][y]
            while curr is not None and curr != state.pawns[player]:
                succ.append(curr)
                (x_, y_) = curr
                curr = prede[x_][y_]
            succ.reverse()
            return succ
        unvisited_succ = [(x_, y_) for (x_, y_) in
                          get_pawn_moves(neighbor) if not visited[x_][y_]]
        for n_ in unvisited_succ:
            (x_, y_) = n_
            if not neighbors.exists(n_):
                neighbors.push((n_, depth + 1), abs(n_[0] - state.goals[player]) + depth + 1)
                prede[x_][y_] = neighbor

    raise NoPath()


def min_steps_before_victory_aStar(state, player):
    """Returns the minimum number of pawn moves necessary for the
    player to reach its goal raw.
    """
    return len(get_shortest_path_aStar(state, player))


def min_steps_before_victory_aStar_simplified(state, player):
    """Returns the minimum number of pawn moves necessary for the
    player to reach its goal raw.
    """
    return len(get_simplified_shortest_path_aStar(state, player))

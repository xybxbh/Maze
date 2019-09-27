import numpy as np
from collections import deque
import queue

import matplotlib.pyplot as plt


class Maze(object):
    class SolutionParams(object):   # storing return values of all search functions
        def __init__(self, has_path = False):
            self.has_path = has_path

        def put(self, has_path, path, max_fringe_size, nodes_expanded):
            self.has_path = has_path
            self.path = path
            self.max_fringe_size = max_fringe_size
            self.nodes_expanded = nodes_expanded

    def __init__(self, p, d, import_m=None):
        if import_m:
            self.dim = d
            self.env = [[import_m.env[row][col] for col in range(d)] for row in range(d)]
        else:
            self.occ_rate = p
            self.dim = d
            self.env = [[np.random.binomial(1, p) for col in range(d)] for row in range(d)]
            # self.env = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
            self.env[0][0] = 0
            self.env[d - 1][d - 1] = 0

    def backtrace(self, path, start):
        # backtrace the path, reference: https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search
        path_list = [(self.dim - 1, self.dim - 1)]
        while path_list[-1] != start:
            path_list.append(path[path_list[-1]])
        path_list.reverse()
        return path_list

    def get_valid(self, cur_x, cur_y, path):
        res = []
        # valid: in the range, no obstruction, not visited
        if cur_x - 1 >= 0 and self.env[cur_x - 1][cur_y] == 0 and (cur_x - 1, cur_y) not in path:
            res.append((cur_x - 1, cur_y))
        if cur_y - 1 >= 0 and self.env[cur_x][cur_y - 1] == 0 and (cur_x, cur_y - 1) not in path:
            res.append((cur_x, cur_y - 1))
        if cur_y + 1 < self.dim and self.env[cur_x][cur_y + 1] == 0 and (cur_x, cur_y + 1) not in path:
            res.append((cur_x, cur_y + 1))
        if cur_x + 1 < self.dim and self.env[cur_x + 1][cur_y] == 0 and (cur_x + 1, cur_y) not in path:
            res.append((cur_x + 1, cur_y))
        return res

    def bfs_solution(self):
        fringe = deque([(0, 0)])
        path = {}
        max_fringe_size = 1
        nodes_expanded = []
        solution_params = self.SolutionParams()
        while fringe:
            if len(fringe) > max_fringe_size:
                max_fringe_size = len(fringe)
            (cur_x, cur_y) = fringe.popleft()
            if cur_x == self.dim - 1 and cur_y == self.dim - 1:
                solution_params.put(True, self.backtrace(path, (0, 0)), max_fringe_size, nodes_expanded)
                return solution_params
            nodes_expanded.append((cur_x, cur_y))
            res = self.get_valid(cur_x, cur_y, path)
            if res:
                for node in res:
                    path[node] = (cur_x, cur_y)
                    fringe.append(node)
        return solution_params

    def dfs_solution(self):  # return hasPath: bool; path: list; max_fringe_size: int
        fringe = [(0, 0)]
        path = {}
        max_fringe_size = 1
        nodes_expanded = []
        solution_params = self.SolutionParams()
        while fringe:
            if len(fringe) > max_fringe_size:
                max_fringe_size = len(fringe)
            (cur_x, cur_y) = fringe.pop()
            if cur_x == self.dim - 1 and cur_y == self.dim - 1:
                solution_params.put(True, self.backtrace(path, (0, 0)), max_fringe_size, nodes_expanded)
                return solution_params
            nodes_expanded.append((cur_x, cur_y))
            res = self.get_valid(cur_x, cur_y, path)
            if res:
                for node in res:
                    path[node] = (cur_x, cur_y)
                    fringe.append(node)
        return solution_params

    def bd_bfs_solution(self):
        """
        fringe: the queue we use when performing bfs
        path(i, j): stores the previous point we visit before step into (i, j)
        visited_node: the point already visited by either side of bfs
        node_expanded: all the point we visited in the algorithm
        node_meet: the point where two bfs meet
        """
        fringe_s = deque([(0, 0)])
        fringe_g = deque([(self.dim - 1, self.dim - 1)])
        path_s = {}
        path_g = {}
        visited_node = [[0 for col in range(self.dim)] for row in range(self.dim)]
        node_expanded = []
        node_meet = (0, 0)
        path_s[(0, 0)] = (0, 0)
        path_g[(self.dim - 1, self.dim - 1)] = (self.dim - 1, self.dim - 1)
        solution_params = self.SolutionParams()
        while fringe_s and fringe_g:
            (cur_x_s, cur_y_s) = fringe_s.popleft()
            (cur_x_g, cur_y_g) = fringe_g.popleft()

            if cur_x_s + 1 < self.dim and self.env[cur_x_s + 1][cur_y_s] == 0 and (cur_x_s + 1, cur_y_s) not in path_s:
                path_s[(cur_x_s + 1, cur_y_s)] = (cur_x_s, cur_y_s)
                if visited_node[cur_x_s + 1][cur_y_s] == 0:
                    fringe_s.append((cur_x_s + 1, cur_y_s))
                    node_expanded.append((cur_x_s + 1, cur_y_s))
                    visited_node[cur_x_s + 1][cur_y_s] = 1
                else:
                    node_meet = (cur_x_s + 1, cur_y_s)
                    break
            if cur_y_s + 1 < self.dim and self.env[cur_x_s][cur_y_s + 1] == 0 and (cur_x_s, cur_y_s + 1) not in path_s:
                path_s[(cur_x_s, cur_y_s + 1)] = (cur_x_s, cur_y_s)
                if visited_node[cur_x_s][cur_y_s + 1] == 0:
                    fringe_s.append((cur_x_s, cur_y_s + 1))
                    node_expanded.append((cur_x_s, cur_y_s + 1))
                    visited_node[cur_x_s][cur_y_s + 1] = 1
                else:
                    node_meet = (cur_x_s, cur_y_s + 1)
                    break
            if cur_x_s - 1 >= 0 and self.env[cur_x_s - 1][cur_y_s] == 0 and (cur_x_s - 1, cur_y_s) not in path_s:
                path_s[(cur_x_s - 1, cur_y_s)] = (cur_x_s, cur_y_s)
                if visited_node[cur_x_s - 1][cur_y_s] == 0:
                    fringe_s.append((cur_x_s - 1, cur_y_s))
                    node_expanded.append((cur_x_s - 1, cur_y_s))
                    visited_node[cur_x_s - 1][cur_y_s] = 1
                else:
                    node_meet = (cur_x_s - 1, cur_y_s)
                    break
            if cur_y_s - 1 >= 0 and self.env[cur_x_s][cur_y_s - 1] == 0 and (cur_x_s, cur_y_s - 1) not in path_s:
                path_s[(cur_x_s, cur_y_s - 1)] = (cur_x_s, cur_y_s)
                if visited_node[cur_x_s][cur_y_s - 1] == 0:
                    fringe_s.append((cur_x_s, cur_y_s - 1))
                    node_expanded.append((cur_x_s, cur_y_s - 1))
                    visited_node[cur_x_s][cur_y_s - 1] = 1
                else:
                    node_meet = (cur_x_s, cur_y_s - 1)
                    break

            if cur_x_g - 1 >= 0 and self.env[cur_x_g - 1][cur_y_g] == 0 and (cur_x_g - 1, cur_y_g) not in path_g:
                path_g[(cur_x_g - 1, cur_y_g)] = (cur_x_g, cur_y_g)
                if visited_node[cur_x_g - 1][cur_y_g] == 0:
                    fringe_g.append((cur_x_g - 1, cur_y_g))
                    node_expanded.append((cur_x_g - 1, cur_y_g))
                    visited_node[cur_x_g - 1][cur_y_g] = 1
                else:
                    node_meet = (cur_x_g - 1, cur_y_g)
                    break
            if cur_y_g - 1 >= 0 and self.env[cur_x_g][cur_y_g - 1] == 0 and (cur_x_g, cur_y_g - 1) not in path_g:
                path_g[(cur_x_g, cur_y_g - 1)] = (cur_x_g, cur_y_g)
                if visited_node[cur_x_g][cur_y_g - 1] == 0:
                    fringe_g.append((cur_x_g, cur_y_g - 1))
                    node_expanded.append((cur_x_g, cur_y_g - 1))
                    visited_node[cur_x_g][cur_y_g - 1] = 1
                else:
                    node_meet = (cur_x_g, cur_y_g - 1)
                    break
            if cur_x_g + 1 < self.dim and self.env[cur_x_g + 1][cur_y_g] == 0 and (cur_x_g + 1, cur_y_g) not in path_g:
                path_g[(cur_x_g + 1, cur_y_g)] = (cur_x_g, cur_y_g)
                if visited_node[cur_x_g + 1][cur_y_g] == 0:
                    fringe_g.append((cur_x_g + 1, cur_y_g))
                    node_expanded.append((cur_x_g + 1, cur_y_g))
                    visited_node[cur_x_g + 1][cur_y_g] = 1
                else:
                    node_meet = (cur_x_g + 1, cur_y_g)
                    break
            if cur_y_g + 1 < self.dim and self.env[cur_x_g][cur_y_g + 1] == 0 and (cur_x_g, cur_y_g + 1) not in path_g:
                path_g[(cur_x_g, cur_y_g + 1)] = (cur_x_g, cur_y_g)
                if visited_node[cur_x_g][cur_y_g + 1] == 0:
                    fringe_g.append((cur_x_g, cur_y_g + 1))
                    node_expanded.append((cur_x_g, cur_y_g + 1))
                    visited_node[cur_x_g][cur_y_g + 1] = 1
                else:
                    node_meet = (cur_x_g, cur_y_g + 1)
                    break

        # if the algorithm finds a solution of the maze
        if node_meet is not (0, 0):
            path_list1 = [node_meet]
            while path_list1[-1] != (0, 0):
                path_list1.append(path_s[path_list1[-1]])
            path_list1.reverse()
            path_list2 = [node_meet]
            while path_list2[-1] != (self.dim - 1, self.dim - 1):
                path_list2.append(path_g[path_list2[-1]])
            path_list1.pop()
            solution_params.put(True, path_list1 + path_list2, 0, node_expanded)
            return solution_params
        else:
            return solution_params

    @staticmethod
    def hf_manhattan(node1, node2):
        (x1, y1) = node1
        (x2, y2) = node2
        return abs(x2 - x1) + abs(y2 - y1)

    @staticmethod
    def hf_euclidean(node1, node2):
        (x1, y1) = node1
        (x2, y2) = node2
        return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    def hf_choose(self, function, node1, node2):
        if function == "Manhattan":
            return self.hf_manhattan(node1, node2)
        elif function == "Euclidean":
            return self.hf_euclidean(node1, node2)
        return False

    def astarsearch_solution(self, heuristicFunction, start):
        fringe = queue.PriorityQueue()
        fringe.put((0, 0, start))
        path = {}
        goal = (self.dim - 1, self.dim - 1)
        max_fringe_size = 1
        nodes_expanded = []
        solution_params = self.SolutionParams(False)
        while not fringe.empty():
            if fringe.qsize() > max_fringe_size:
                max_fringe_size = fringe.qsize()
            (total_estCost, alr_cost, (cur_x, cur_y)) = fringe.get()
            if (cur_x, cur_y) == goal:
                # print(self.backtrace(path, start))
                solution_params.put(True, self.backtrace(path, start), max_fringe_size, nodes_expanded)
                return solution_params
            nodes_expanded.append((cur_x, cur_y))
            res = self.get_valid(cur_x, cur_y, path)
            if res:
                for node in res:
                    path[node] = (cur_x, cur_y)
                    est_cost = self.hf_choose(heuristicFunction, node, goal)
                    fringe.put((est_cost - alr_cost + 1, alr_cost - 1, node))
        return solution_params

    def solve(self, alg, heuristicFunction="Manhattan", start=(0, 0)):
        if alg == "dfs":
            return self.dfs_solution()
        elif alg == "bfs":
            return self.bfs_solution()
        elif alg == "bdbfs":
            return self.bd_bfs_solution()
        elif alg == "a*":
            return self.astarsearch_solution(heuristicFunction, start)
        else:
            return False

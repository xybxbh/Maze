import numpy as np
from collections import deque
import queue

import matplotlib.pyplot as plt


class Maze(object):
    class SolutionParams(object):
        def __init__(self, has_path):
            self.has_path = has_path

        def dfs(self, has_path, path, max_fringe_size):
            self.has_path = has_path
            self.path = path
            self.max_fringe_size = max_fringe_size

        def aStar(self, has_path, path, max_nodes_expanded):
            self.has_path = has_path
            self.path = path
            self.max_nodes_expanded = max_nodes_expanded

    def __init__(self, p, d, import_m=None):
        if import_m:
            self.dim = d
            self.env = [[import_m.env[row][col] for col in range(d)] for row in range(d)]
        else:
            self.occ_rate = p
            self.dim = d
            # self.env = [[np.random.binomial(1, p) for col in range(d)] for row in range(d)]
            self.env = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
            # self.env = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
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
        while fringe:
            (cur_x, cur_y) = fringe.popleft()
            if cur_x == self.dim - 1 and cur_y == self.dim - 1:
                return self.backtrace(path, (0, 0))
            res = self.get_valid(cur_x, cur_y, path)
            if res:
                for node in res:
                    path[node] = (cur_x, cur_y)
                    fringe.append(node)
        return False

    def dfs_solution(self):  # return hasPath: bool; path: list; max_fringe_size: int
        fringe = [(0, 0)]
        path = {}
        max_fringe_size = 1
        solution_params = self.SolutionParams(False)
        while fringe:
            if len(fringe) > max_fringe_size:
                max_fringe_size = len(fringe)
            (cur_x, cur_y) = fringe.pop()
            if cur_x == self.dim - 1 and cur_y == self.dim - 1:
                solution_params.dfs(True, self.backtrace(path, (0, 0)), max_fringe_size)
                return solution_params
            res = self.get_valid(cur_x, cur_y, path)
            if res:
                for node in res:
                    path[node] = (cur_x, cur_y)
                    fringe.append(node)
        return solution_params

    def bd_dfs_solution(self):
        fringe_s = [(0, 0)]
        fringe_g = [(self.dim - 1, self.dim - 1)]
        path_s = {}
        path_g = {}
        visited_s = [(0, 0)]
        visited_g = [(self.dim - 1, self.dim - 1)]
        finish = False
        path_s[(0, 0)] = (0, 0)
        path_g[(self.dim - 1, self.dim - 1)] = (self.dim - 1, self.dim - 1)
        while fringe_s and fringe_g:
            (cur_x_s, cur_y_s) = fringe_s.pop()
            (cur_x_g, cur_y_g) = fringe_g.pop()

            if cur_x_s + 1 < self.dim and self.env[cur_x_s + 1][cur_y_s] == 0 and (cur_x_s + 1, cur_y_s) not in path_s:
                path_s[(cur_x_s + 1, cur_y_s)] = (cur_x_s, cur_y_s)
                fringe_s.append((cur_x_s + 1, cur_y_s))
                visited_s.append((cur_x_s + 1, cur_y_s))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break
            if cur_x_s - 1 >= 0 and self.env[cur_x_s - 1][cur_y_s] == 0 and (cur_x_s - 1, cur_y_s) not in path_s:
                path_s[(cur_x_s - 1, cur_y_s)] = (cur_x_s, cur_y_s)
                fringe_s.append((cur_x_s - 1, cur_y_s))
                visited_s.append((cur_x_s - 1, cur_y_s))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break
            if cur_y_s + 1 < self.dim and self.env[cur_x_s][cur_y_s + 1] == 0 and (cur_x_s, cur_y_s + 1) not in path_s:
                path_s[(cur_x_s, cur_y_s + 1)] = (cur_x_s, cur_y_s)
                fringe_s.append((cur_x_s, cur_y_s + 1))
                visited_s.append((cur_x_s, cur_y_s + 1))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break
            if cur_y_s - 1 >= 0 and self.env[cur_x_s][cur_y_s - 1] == 0 and (cur_x_s, cur_y_s - 1) not in path_s:
                path_s[(cur_x_s, cur_y_s - 1)] = (cur_x_s, cur_y_s)
                fringe_s.append((cur_x_s, cur_y_s - 1))
                visited_s.append((cur_x_s, cur_y_s - 1))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break

            if cur_x_g + 1 < self.dim and self.env[cur_x_g + 1][cur_y_g] == 0 and (cur_x_g + 1, cur_y_g) not in path_g:
                path_g[(cur_x_g + 1, cur_y_g)] = (cur_x_g, cur_y_g)
                fringe_g.append((cur_x_g + 1, cur_y_g))
                visited_g.append((cur_x_g + 1, cur_y_g))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break
            if cur_x_g - 1 >= 0 and self.env[cur_x_g - 1][cur_y_g] == 0 and (cur_x_g - 1, cur_y_g) not in path_g:
                path_g[(cur_x_g - 1, cur_y_g)] = (cur_x_g, cur_y_g)
                fringe_g.append((cur_x_g - 1, cur_y_g))
                visited_g.append((cur_x_g - 1, cur_y_g))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break
            if cur_y_g + 1 < self.dim and self.env[cur_x_g][cur_y_g + 1] == 0 and (cur_x_g, cur_y_g + 1) not in path_g:
                path_g[(cur_x_g, cur_y_g + 1)] = (cur_x_g, cur_y_g)
                fringe_g.append((cur_x_g, cur_y_g + 1))
                visited_g.append((cur_x_g, cur_y_g + 1))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break
            if cur_y_g - 1 >= 0 and self.env[cur_x_g][cur_y_g - 1] == 0 and (cur_x_g, cur_y_g - 1) not in path_g:
                path_g[(cur_x_g, cur_y_g - 1)] = (cur_x_g, cur_y_g)
                fringe_g.append((cur_x_g, cur_y_g - 1))
                visited_g.append((cur_x_g, cur_y_g - 1))
                if list(set(visited_s).intersection(set(visited_g))):
                    finish = True
                    break

        if finish:
            path_list1 = list(set(visited_s).intersection(set(visited_g)))
            while path_list1[-1] != (0, 0):
                path_list1.append(path_s[path_list1[-1]])
            path_list1.reverse()
            path_list2 = list(set(visited_s).intersection(set(visited_g)))
            while path_list2[-1] != (self.dim - 1, self.dim - 1):
                path_list2.append(path_g[path_list2[-1]])
            path_list1.pop()
            return path_list1 + path_list2
        else:
            return False

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
        max_nodes_expanded = 0
        solution_params = self.SolutionParams(False)
        while not fringe.empty():
            (total_estCost, alr_cost, (cur_x, cur_y)) = fringe.get()
            if (cur_x, cur_y) == goal:
                # print(self.backtrace(path, start))
                solution_params.aStar(True, self.backtrace(path, start), max_nodes_expanded)
                return solution_params
            max_nodes_expanded += 1
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
            return self.bd_dfs_solution()
        elif alg == "a*":
            return self.astarsearch_solution(heuristicFunction, start)
        else:
            return False


class TestMaze(object):

    def __init__(self, p, d, input_maze=None):
        self.maze = Maze(p, d, input_maze)
        # self.test_init()
        # self.test_bfs()
        # self.test_dfs()
        self.test_astarManh()
        # self.test_astarEucl()
        # self.test_bddfs()

    def test_init(self):
        print(self.maze.env)

    def test_dfs(self):
        param = self.maze.dfs_solution()
        if param.has_path:
            print(param.path)
            self.printGraph(param.path)
        else:
            self.printGraph(param.has_path)

    def test_bfs(self):
        path = self.maze.bfs_solution()
        self.printGraph(path)

    def test_bddfs(self):
        path = self.maze.bd_dfs_solution()
        self.printGraph(path)

    def test_astarEucl(self):
        path = self.maze.astarsearch_solution("Euclidean")
        self.printGraph(path)

    def test_astarManh(self):
        path1 = self.maze.solve("a*")
        print(path1.max_nodes_expaned)
        self.printGraph(path1.path)

    def printGraph(self, path):
        print(path)
        plt.xlim(0, self.maze.dim)
        plt.ylim(0, self.maze.dim)
        plt.xticks(range(0, self.maze.dim, 1))
        plt.yticks(range(0, self.maze.dim, 1))
        plt.grid(True, linestyle="-", color="black", linewidth="0.5")
        axes = plt.gca()
        axes.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

        data = np.array(self.maze.env)
        rows, cols = data.shape

        axeslist_x = []
        axeslist_y = []

        if path == False:
            plt.title("NO SOLUTION")
        else:
            for t in path:
                (y, x) = t
                axeslist_x.append(x + 0.5)
                axeslist_y.append(self.maze.dim - y - 0.5)
            plt.plot(axeslist_x, axeslist_y, c="red")
        plt.imshow(data, cmap='gray_r', extent=[0, cols, 0, rows])

        plt.show()


if __name__ == "__main__":
    TestMaze(0.2, 30)


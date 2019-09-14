import numpy as np
from collections import deque

class Maze(object):
    def __init__(self, p, d):
        self.occ_p = p
        self.dim = d
        self.env = [[np.random.binomial(1, p) for col in range(d)] for row in range(d)]
        self.env[0][0] = 0
        self.env[d - 1][d - 1] = 0

    def bfs_solution(self):
        nodes = deque([(0, 0)])
        path = {}
        while nodes:
            (cur_x, cur_y) = nodes.popleft()
            if cur_x == self.dim - 1 and cur_y == self.dim - 1:
                # backtrace, reference: https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search
                path_list = [(self.dim - 1, self.dim - 1)]
                while path_list[-1] != (0, 0):
                    path_list.append(path[path_list[-1]])
                path_list.reverse()
                return path_list
            # valid: in the range, no obstruction, not visited
            if cur_x + 1 < self.dim and self.env[cur_x + 1][cur_y] == 0 and (cur_x + 1, cur_y) not in path:
                path[(cur_x + 1, cur_y)] = (cur_x, cur_y)
                nodes.append((cur_x + 1, cur_y))
            if cur_x - 1 >= 0 and self.env[cur_x - 1][cur_y] == 0 and (cur_x - 1, cur_y) not in path:
                path[(cur_x - 1, cur_y - 1)] = (cur_x, cur_y)
                nodes.append((cur_x - 1, cur_y))
            if cur_y + 1 < self.dim and self.env[cur_x][cur_y + 1] == 0 and (cur_x, cur_y + 1) not in path:
                path[(cur_x, cur_y + 1)] = (cur_x, cur_y)
                nodes.append((cur_x, cur_y + 1))
            if cur_y - 1 >= 0 and self.env[cur_x][cur_y - 1] == 0 and (cur_x, cur_y - 1) not in path:
                path[(cur_x, cur_y - 1)] = (cur_x, cur_y)
                nodes.append((cur_x, cur_y - 1))
        return False

    def dfs_solution(self):
        nodes = [(0, 0)]
        path = {}
        while nodes:
            (cur_x, cur_y) = nodes.pop()
            if cur_x == self.dim - 1 and cur_y == self.dim - 1:
                path_list = [(self.dim - 1, self.dim - 1)]
                while path_list[-1] != (0, 0):
                    path_list.append(path[path_list[-1]])
                path_list.reverse()
                return path_list
            if cur_x + 1 < self.dim and self.env[cur_x + 1][cur_y] == 0 and (cur_x + 1, cur_y) not in path:
                path[(cur_x + 1, cur_y)] = (cur_x, cur_y)
                nodes.append((cur_x + 1, cur_y))
            if cur_x - 1 >= 0 and self.env[cur_x - 1][cur_y] == 0 and (cur_x - 1, cur_y) not in path:
                path[(cur_x - 1, cur_y - 1)] = (cur_x, cur_y)
                nodes.append((cur_x - 1, cur_y))
            if cur_y + 1 < self.dim and self.env[cur_x][cur_y + 1] == 0 and (cur_x, cur_y + 1) not in path:
                path[(cur_x, cur_y + 1)] = (cur_x, cur_y)
                nodes.append((cur_x, cur_y + 1))
            if cur_y - 1 >= 0 and self.env[cur_x][cur_y - 1] == 0 and (cur_x, cur_y - 1) not in path:
                path[(cur_x, cur_y - 1)] = (cur_x, cur_y)
                nodes.append((cur_x, cur_y - 1))
        return False

if __name__ == "__main__":
    maze = Maze(0.1, 10)
    print(maze.env)
    if maze.dfs_solution() == False:
        print('pass')
    else:
        for t in maze.dfs_solution():
            print(t)

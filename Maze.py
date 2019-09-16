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

    def bd_bfs_solution(self):
        nodes_s = deque([(0, 0)])
        nodes_g = deque([(self.dim - 1, self.dim - 1)])
        path_s = {}
        path_g = {}
        visited_s = [(0, 0)]
        visited_g = [(self.dim - 1, self.dim - 1)]
        finish = False
        path_s[(0, 0)] = (0, 0)
        path_g[(self.dim - 1, self.dim - 1)] = (self.dim - 1, self.dim - 1)
        while nodes_s and nodes_g:
            (cur_x_s, cur_y_s) = nodes_s.popleft()
            (cur_x_g, cur_y_g) = nodes_g.popleft()

            if cur_x_s + 1 < self.dim and self.env[cur_x_s + 1][cur_y_s] == 0 and (cur_x_s + 1, cur_y_s) not in path_s:
                path_s[(cur_x_s + 1, cur_y_s)] = (cur_x_s, cur_y_s)
                nodes_s.append((cur_x_s + 1, cur_y_s))
                visited_s.append((cur_x_s + 1, cur_y_s))
            if cur_x_s - 1 >= 0 and self.env[cur_x_s - 1][cur_y_s] == 0 and (cur_x_s - 1, cur_y_s) not in path_s:
                path_s[(cur_x_s - 1, cur_y_s)] = (cur_x_s, cur_y_s)
                nodes_s.append((cur_x_s - 1, cur_y_s))
                visited_s.append((cur_x_s - 1, cur_y_s))
            if cur_y_s + 1 < self.dim and self.env[cur_x_s][cur_y_s + 1] == 0 and (cur_x_s, cur_y_s + 1) not in path_s:
                path_s[(cur_x_s, cur_y_s + 1)] = (cur_x_s, cur_y_s)
                nodes_s.append((cur_x_s, cur_y_s + 1))
                visited_s.append((cur_x_s, cur_y_s + 1))
            if cur_y_s - 1 >= 0 and self.env[cur_x_s][cur_y_s - 1] == 0 and (cur_x_s, cur_y_s - 1) not in path_s:
                path_s[(cur_x_s, cur_y_s - 1)] = (cur_x_s, cur_y_s)
                nodes_s.append((cur_x_s, cur_y_s - 1))
                visited_s.append((cur_x_s, cur_y_s - 1))
                
            if list(set(visited_s).intersection(set(visited_g))):
                finish = True
                break

            if cur_x_g + 1 < self.dim and self.env[cur_x_g + 1][cur_y_g] == 0 and (cur_x_g + 1, cur_y_g) not in path_g:
                path_g[(cur_x_g + 1, cur_y_g)] = (cur_x_g, cur_y_g)
                nodes_g.append((cur_x_g + 1, cur_y_g))
                visited_g.append((cur_x_g + 1, cur_y_g))
            if cur_x_g - 1 >= 0 and self.env[cur_x_g - 1][cur_y_g] == 0 and (cur_x_g - 1, cur_y_g) not in path_g:
                path_g[(cur_x_g - 1, cur_y_g)] = (cur_x_g, cur_y_g)
                nodes_g.append((cur_x_g - 1, cur_y_g))
                visited_g.append((cur_x_g - 1, cur_y_g))
            if cur_y_g + 1 < self.dim and self.env[cur_x_g][cur_y_g + 1] == 0 and (cur_x_g, cur_y_g + 1) not in path_g:
                path_g[(cur_x_g, cur_y_g + 1)] = (cur_x_g, cur_y_g)
                nodes_g.append((cur_x_g, cur_y_g + 1))
                visited_g.append((cur_x_g, cur_y_g + 1))
            if cur_y_g - 1 >= 0 and self.env[cur_x_g][cur_y_g - 1] == 0 and (cur_x_g, cur_y_g - 1) not in path_g:
                path_g[(cur_x_g, cur_y_g - 1)] = (cur_x_g, cur_y_g)
                nodes_g.append((cur_x_g, cur_y_g - 1))
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

            return path_list1 + path_list2
        else:
            return False

if __name__ == "__main__":
    maze = Maze(0.1, 10)
    print(maze.env)
    if maze.bd_bfs_solution() == False:
        print('pass')
    else:
        for t in maze.bd_bfs_solution():
            print(t)


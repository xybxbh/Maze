import random
from copy import deepcopy
from Maze import *
import math

class LocalSearch(object):
    def __init__(self, ori_maze):
        self.ori_maze = ori_maze
        self.cur_maze = ori_maze
        while self.cur_maze.solve("dfs") == False:
            regen_maze = Maze(ori_maze.occ_p, ori_maze.dim)
            self.cur_maze = regen_maze
        self.sa_tem = 100
        self.sa_tmin = 1e-8
        self.sa_delta = 0.98
    
    def update_maze_param(self, maze):
        '''self.env = self.cur_maze.env
        self.dim = self.cur_maze.dim
        self.occ_p = self.cur_maze.occ_p'''
        self.cur_maze = maze
        self.num_ob = self.get_obstruction_num()

    def get_obstruction_num(self):
        count = 0
        for i in range(self.cur_maze.dim):
            for j in range(self.cur_maze.dim):
                if self.cur_maze.env[i][j] == 1:
                    count += 1
        return count

    def gen_maze(self):
        # one random neighbor in each iteration
        p = random.random()
        new_maze = deepcopy(self.cur_maze)
        while(True):
            x = random.randint(0, self.cur_maze.dim - 1)
            y = random.randint(0, self.cur_maze.dim - 1)
            if self.cur_maze.env[x][y] == 0:
                new_maze.env[x][y] = 1
                if not new_maze.solve("dfs") == False:
                    break
            elif self.cur_maze.env[x][y] == 1 and p < self.num_ob/(self.cur_maze.dim*self.cur_maze.dim):
                new_maze.env[x][y] = 0
                if not new_maze.solve("dfs") == False:
                    break
        return new_maze

    def comparator(self, alg, maze):
        if alg == 'dfs':
            path_new, max_fringe_size_new = maze.dfs_solution()
            path_cur, max_fringe_size_cur = self.cur_maze.dfs_solution()
            return max_fringe_size_new - max_fringe_size_cur
        return

    def simulated_annealing(self, alg):
        while self.sa_tem > self.sa_tmin:
            # here we can do search with multi-neighbors
            new_maze = self.gen_maze()
            p = random.random()
            comparator = self.comparator(alg, new_maze)
            if comparator > 0:
                self.update_maze_param(new_maze)
            elif comparator <= 0 and p < 1/(1 + math.exp(0 - comparator/self.sa_tem)):
                self.update_maze_param(new_maze)
            self.sa_tem = self.sa_tem * self.sa_delta
            print('iterating')

if __name__ == "__main__":
    ori_maze = Maze(0.2, 10)
    sa = LocalSearch(ori_maze)
    sa.simulated_annealing("dfs")
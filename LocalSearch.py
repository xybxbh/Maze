import random
from copy import deepcopy
from Maze import *
import math
import numpy as np

class LocalSearch(object):
    def __init__(self, ori_maze):
        self.ori_maze = ori_maze
        self.update_maze_param(ori_maze)
        while self.cur_maze.solve("dfs").has_path == False:
            print('false')
            regen_maze = Maze(ori_maze.occ_p, ori_maze.dim)
            self.update_maze_param(regen_maze)
        self.sa_tem = 1
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
        it = 0
        new_maze = deepcopy(self.cur_maze)
        while(True):
            x = random.randint(0, self.cur_maze.dim - 1)
            y = random.randint(0, self.cur_maze.dim - 1)
            if self.cur_maze.env[x][y] == 0:
                new_maze.env[x][y] = 1
                if new_maze.solve("dfs").has_path == False:
                    it += 1
                else:
                    break
            elif self.cur_maze.env[x][y] == 1 and p < self.num_ob/(self.cur_maze.dim*self.cur_maze.dim):
                new_maze.env[x][y] = 0
                if new_maze.solve("dfs").has_path == False:
                    it += 1
                else:
                    break
            else:
                print(x, y)
                it += 1
            if it == 1000:
                print('iter protection')
                return False
        return new_maze

    def comparator(self, alg, maze):
        params_new = maze.solve(alg)
        params_cur = self.cur_maze.solve(alg)
        return params_new.max_fringe_size - params_new.max_fringe_size

    def simulated_annealing(self, alg):
        while self.sa_tem > self.sa_tmin:
            # here we can do search with multi-neighbors
            print('iterating')
            new_maze = self.gen_maze()
            if new_maze == False:
                continue
            p = random.random()
            comparator = self.comparator(alg, new_maze)
            p_sa = 1/(1 + math.exp(0 - comparator/self.sa_tem)) if (0 - comparator/self.sa_tem) < 500 else 0
            if comparator > 0 or p < p_sa:
                self.update_maze_param(new_maze)
                print('comparator =', comparator)
                print('p =', p)
                print('psa =', p_sa)
            self.sa_tem = self.sa_tem * self.sa_delta
            # print('t =', self.sa_tem)

if __name__ == "__main__":
    ori_maze = Maze(0.3, 100)
    print('loop')
    sa = LocalSearch(ori_maze)
    sa.simulated_annealing("dfs")
    print(sa.num_ob)
    print(sa.cur_maze.solve('dfs').max_fringe_size)
    print(len(sa.cur_maze.solve('dfs').path))
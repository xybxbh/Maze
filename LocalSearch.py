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

        self.population = 1000
        self.genetic_iter = 100
        self.breed_cnt = 1000
        self.mutant_rate = 0.6

    
    def update_maze_param(self, maze):
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
        p_cutpath = random.random()
        p_o2z = random.random()
        it = 0
        new_maze = deepcopy(self.cur_maze)
        while(True):
            if p_cutpath < 0.6:
                path = self.cur_maze.solve('dfs').path
                index = random.randint(1, len(path) - 2)
                (x_cut, y_cut) = path[index]
                new_maze.env[x_cut][y_cut] = 1
                break
            x = random.randint(0, self.cur_maze.dim - 1)
            y = random.randint(0, self.cur_maze.dim - 1)
            if (x == 0 and y == 0) or (x == self.cur_maze.dim - 1 and y == self.cur_maze.dim - 1):
                continue
            if self.cur_maze.env[x][y] == 0 or p_o2z < self.num_ob/(self.cur_maze.dim*self.cur_maze.dim):
                new_maze.env[x][y] = 1 - new_maze.env[x][y]
                # solvablility certification in the sa function
                break
            else:
                print(x, y)
                it += 1
            if it == 1000:
                print('iter protection')
                break
        return new_maze

    def comparator(self, alg, param, maze):
        params_new = maze.solve(alg)
        params_cur = self.cur_maze.solve(alg)
        if param == 'max_fringe_size':
            return params_new.max_fringe_size - params_cur.max_fringe_size
        return False

    def simulated_annealing(self, alg):
        while self.sa_tem > self.sa_tmin:
            # here we can do search with multi-neighbors
            new_maze = self.gen_maze()
            if not new_maze.solve('dfs').has_path:
                continue
            p = random.random()
            comparator = self.comparator(alg, 'max_fringe_size', new_maze)
            p_sa = 1/(1 + math.exp(0 - comparator/self.sa_tem)) if (0 - comparator/self.sa_tem) < 500 else 0
            if comparator > 0 or p < p_sa:
                self.update_maze_param(new_maze)
                print('comparator =', comparator)
                print(sa.cur_maze.solve('dfs').max_fringe_size)
                print(len(self.cur_maze.solve('dfs').path))
            self.sa_tem = self.sa_tem * self.sa_delta
            #print('t =', self.sa_tem)

    def Maze_Generate(self, d):
        p = random.uniform(0.3, 0.4)
        tmp_m = Maze(p, d)
        while not tmp_m.solve('dfs').has_path:
            tmp_m = Maze(p, d)
        return tmp_m

    def dfs(self, m, dim):
        fringe = deque([(0, 0)])
        path = {}
        while fringe:
            (cur_x, cur_y) = fringe.popleft()
            if cur_x == dim - 1 and cur_y == dim - 1:
                path_list = [(dim - 1, dim - 1)]
                while path_list[-1] != (0, 0):
                    path_list.append(path[path_list[-1]])
                path_list.reverse()
                return len(path_list)
            res = []
            # valid: in the range, no obstruction, not visited
            if cur_x + 1 < dim and m[cur_x + 1][cur_y] == 0 and (cur_x + 1, cur_y) not in path:
                res.append((cur_x + 1, cur_y))
            if cur_x - 1 >= 0 and m[cur_x - 1][cur_y] == 0 and (cur_x - 1, cur_y) not in path:
                res.append((cur_x - 1, cur_y))
            if cur_y + 1 < dim and m[cur_x][cur_y + 1] == 0 and (cur_x, cur_y + 1) not in path:
                res.append((cur_x, cur_y + 1))
            if cur_y - 1 >= 0 and m[cur_x][cur_y - 1] == 0 and (cur_x, cur_y - 1) not in path:
                res.append((cur_x, cur_y - 1))
            if res:
                for node in res:
                    path[node] = (cur_x, cur_y)
                    fringe.append(node)
        return False

    def Maze_Score(self, m, d):
        if m.solve('dfs').has_path is False:
            return ori_maze.dim
        else:
            return len(m.solve('dfs').path)

    def Maze_merge(self, m1, m2, d, strategy):
        m1_n = deepcopy(m1)
        m2_n = deepcopy(m2)
        if strategy == 1:
            m3.env = m1_n.env[0:int(d/2)] + m2_n.env[int(d/2):d]
        elif strategy == 2:
            m3 = m1_n
            for i in range(0, d):
                for j in range(0, d):
                    if i < d / 2:
                        if j < d / 2:
                            m3.env[i][j] = m1_n.env[i][j]
                        else:
                            m3.env[i][j] = m2_n.env[i][j]
                    else:
                        if j < d / 2:
                            m3.env[i][j] = m2_n.env[i][j]
                        else:
                            m3.env[i][j] = m1_n.env[i][j]
        if random.uniform(0, 1) < self.mutant_rate:
            x = random.randint(1, self.cur_maze.dim - 2)
            y = random.randint(1, self.cur_maze.dim - 2)
            m3.env[x][y] = 1 - m3.env[x][y]
        return m3

    def random_picks(self, m_list, weight):
        x = random.uniform(0, sum(weight))
        cumulative_weight = 0.0
        for m_list, m_weight in zip(m_list, weight):
            cumulative_weight += m_weight
            if x < cumulative_weight:
                break
        return m_list

    def genetic_algorithm(self):
        # Generate Parents
        maze_population = []
        for t in range(self.population):
            maze_population.append(deepcopy(self.Maze_Generate(ori_maze.dim)))
        weight = []

        for k in range(self.genetic_iter):
            print("k=", k)
            weight.clear()
            weight = [self.Maze_Score(maze_population[x], ori_maze.dim) for x in range(self.population)]
            # Generate children
            i = 0
            while i < self.breed_cnt:

                m1 = self.random_picks(maze_population, weight)
                m2 = self.random_picks(maze_population, weight)
                while m1 == m2:
                    m2 = self.random_picks(maze_population, weight)
                merge_m = self.Maze_merge(m1, m2, ori_maze.dim, 2)

                if merge_m.solve('dfs').has_path:
                    maze_population.append(deepcopy(merge_m))
                    weight.append(self.Maze_Score(merge_m, ori_maze.dim))
                    self.population += 1
                    i += 1

            # Keep The fittest
            weight_tmp = weight
            min_k = np.partition(np.array(weight_tmp), self.breed_cnt - 1)[self.breed_cnt - 1]
            j = 0
            cnt = self.breed_cnt
            while j < self.population and cnt > 0:
                if weight[j] <= min_k:
                    maze_population.pop(j)
                    weight.pop(j)
                    self.population -= 1
                    cnt -= 1
                j += 1

        best_index = weight.index(max(weight))
        print(maze_population[best_index])
        print(len(maze_population[best_index].solve('dfs').path))
        return maze_population[best_index]

if __name__ == "__main__":
    ori_maze = Maze(0.2, 30)
    sa = LocalSearch(ori_maze)

    maze0 = sa.genetic_algorithm()
    TestMaze(1, 10, maze0)
    
    # sa.simulated_annealing("dfs")
    # print(sa.sa_tem)
    # print(sa.num_ob)
    # print(sa.cur_maze.solve('dfs').max_fringe_size)
    # print(len(sa.cur_maze.solve('dfs').path))
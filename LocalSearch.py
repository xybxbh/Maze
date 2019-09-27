import random
from copy import deepcopy
from Maze import *
import math
import queue
import numpy as np
from VisualizationPath import printGraph

class LocalSearch(object):
    def __init__(self, ori_maze):
        self.ori_maze = ori_maze
        self.update_maze_param(ori_maze)
        while self.cur_maze.solve("dfs").has_path == False:
            print('false')
            regen_maze = Maze(ori_maze.occ_rate, ori_maze.dim)
            self.update_maze_param(regen_maze)
        # simulated annealing params
        self.sa_tem = 1
        self.sa_tmin = 1e-8
        self.sa_delta = 0.98

        # genetic algorithm params
        self.population = 1000
        self.genetic_iter = 2
        self.breed_cnt = 1000
        self.mutant_rate = 0.6


    def update_maze_param(self, maze):  # bound maze update and num_ob update together
        self.cur_maze = maze
        self.num_ob = self.get_obstruction_num()

    def get_obstruction_num(self):
        count = 0
        for i in range(self.cur_maze.dim):
            for j in range(self.cur_maze.dim):
                if self.cur_maze.env[i][j] == 1:
                    count += 1
        return count

    def get_obstruction_num2(self, cur_maze):
        count = 0
        for i in range(cur_maze.dim):
            for j in range(cur_maze.dim):
                if cur_maze.env[i][j] == 1:
                    count += 1
        return count

    def gen_maze(self, path):
        # one random neighbor in each iteration
        p_cutpath = random.random() # prob that cutting a node in the path, otherwise selecting randomly in the maze
        p_o2z = random.random() # prob that setting a node with obstruction to available
        it = 0
        new_maze = deepcopy(self.cur_maze)
        while True:
            if p_cutpath < 0.6:
                # path = self.cur_maze.solve('dfs').path
                index = random.randint(1, len(path) - 2)
                (x_cut, y_cut) = path[index]
                new_maze.env[x_cut][y_cut] = 1
                break
            x = random.randint(0, self.cur_maze.dim - 1)
            y = random.randint(0, self.cur_maze.dim - 1)
            if (x == 0 and y == 0) or (x == self.cur_maze.dim - 1 and y == self.cur_maze.dim - 1):
                continue
            if self.cur_maze.env[x][y] == 0 or p_o2z < self.num_ob / (self.cur_maze.dim * self.cur_maze.dim):
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


    def gen_maze3(self, cur_maze, iter):
        # one random neighbor in each iteration
        p_cutpath = random.random()
        it = 0
        new_maze = deepcopy(cur_maze)
        path_param =  cur_maze.solve("dfs")
        path = path_param.path
        max_fringe_size = path_param.max_fringe_size
        while(True):
            # x, y = path[random.randint(0, len(path) - 1)]
            # new_maze.env[x][y] = 1
            # if new_maze.solve("dfs").has_path == False:
            #     it += 1
            #     new_maze.env[x][y] = 0
            # else:
            #     break
            # if p_cutpath < 0.6:
            if iter < 30:
                path = self.cur_maze.solve('dfs').path
                index = random.randint(1, len(path) - 2)
                (x_cut, y_cut) = path[index]
                new_maze.env[x_cut][y_cut] = 1

            x = random.randint(0, cur_maze.dim - 1)
            y = random.randint(0, cur_maze.dim - 1)

            if (x == 0 and y == 0) or (x == self.cur_maze.dim - 1 and y == self.cur_maze.dim - 1):
                continue
            if cur_maze.env[x][y] == 0:
                new_maze.env[x][y] = 1
                if new_maze.solve("dfs").has_path == False:
                    it += 1
                else:
                    break
            if it == 1000:
                print('iter protection')
                return False
        return new_maze

    def gen_maze4(self, cur_maze, iter):
        # one random neighbor in each iteration
        p = random.random()
        it = 0
        new_maze = deepcopy(cur_maze)
        while(True):
            change = 0
            if iter < 30:
                for i in range(0, 3):
                    x = random.randint(0, cur_maze.dim - 1)
                    y = random.randint(0, cur_maze.dim - 1)
                    if cur_maze.env[x][y] == 0:
                        new_maze.env[x][y] = 1
                        change = 1
            else:
                x = random.randint(0, cur_maze.dim - 1)
                y = random.randint(0, cur_maze.dim - 1)
                if cur_maze.env[x][y] == 0:
                    new_maze.env[x][y] = 1
                    change = 1
            if change == 1:
                if new_maze.solve("dfs").has_path == False:
                    it += 1
                else:
                    break
            # elif cur_maze.env[x][y] == 1 and p < num_ob/(cur_maze.dim * cur_maze.dim):
            #     new_maze.env[x][y] = 0
            #     if new_maze.solve("dfs").has_path == False:
            #         it += 1
            #     else:
            #         break
            else:
                # print(x, y)
                it += 1
            if it == 1000:
                print('iter protection')
                return False
        return new_maze

    def comparator(self, alg, param, maze):
        params_new = maze.solve(alg)
        params_cur = self.cur_maze.solve(alg)
        if param == "max_fringe_size":
            return params_new.max_fringe_size - params_cur.max_fringe_size
        elif param == "total_nodes_expanded":
            return len(params_new.nodes_expanded) - len(params_cur.max_nodes_expaned)
        elif param == "total_path_length":
            return len(params_new.path) - len(params_cur.path)
        return False

    def simulated_annealing(self, alg):
        while self.sa_tem > self.sa_tmin:
            new_maze = self.gen_maze(self.cur_maze.solve(alg).path)
            if not new_maze.solve(alg).has_path:
                continue
            p = random.random()
            comparator = self.comparator(alg, "max_fringe_size", new_maze)
            # e^700+ will be out of stack
            p_sa = 1 / (1 + math.exp(0 - comparator / self.sa_tem)) if (0 - comparator / self.sa_tem) < 500 else 0
            if comparator > 0 or p < p_sa:
                self.update_maze_param(new_maze)
                print('comparator =', comparator)
                print(self.get_search_condition(alg, "max_fringe_size", self.cur_maze.solve(alg)))
                print(len(self.cur_maze.solve(alg).path))
            self.sa_tem = self.sa_tem * self.sa_delta
            # print('t =', self.sa_tem)

    def get_search_condition(self, param_name, solution_param):
        if param_name == "max_fringe_size":
            return solution_param.max_fringe_size
        elif param_name == "total_nodes_expanded":
            return len(solution_param.nodes_expanded)
        elif param_name == "total_path_length":
            return len(solution_param.path)
        return False

    def gen_ini_maze_bs(self, alg, m):
        ini_maze_list = queue.PriorityQueue()
        index = 0
        for i in range(0, m):
            maze = Maze(self.ori_maze.occ_rate, self.ori_maze.dim)
            solution_param = maze.solve(alg)
            while solution_param.has_path == False:
                maze = Maze(self.ori_maze.occ_rate, self.ori_maze.dim)
                solution_param = maze.solve(alg)
            ini_maze_list.put((-self.get_search_condition(alg, "total_nodes_expanded", solution_param), index, maze))
            index += 1
        print("ini_finished")
        return (ini_maze_list, index)

    def beam_search(self, alg, m, k):
        cur_maze_list, index = self.gen_ini_maze_bs(alg, m)
        hard_maze_list = queue.PriorityQueue()
        new_maze_list = queue.PriorityQueue()
        beam_search_iter = 0
        while not cur_maze_list.empty():
            beam_search_iter += 1
            if beam_search_iter == 200:
                break
            print("_________________________beam_search+" + str(beam_search_iter))
            for i in range(0, m):
                if cur_maze_list.empty():
                    break
                temp_maze_list = queue.PriorityQueue()
                cur_max_condition, cur_index, cur_maze = cur_maze_list.get()
                print("cur_best_size: " + str(cur_max_condition))
                for j in range(0, k):
                    new_maze = self.gen_maze4(cur_maze, beam_search_iter)
                    if new_maze:
                        new_path_param = new_maze.solve(alg)
                        if self.get_search_condition(alg, "total_nodes_expanded", new_path_param) <= -cur_max_condition:
                            continue
                        temp_maze_list.put((-self.get_search_condition(alg, new_path_param), index, new_maze))
                        index += 1
                    else:
                        hard_maze_list.put((cur_max_condition, cur_index, cur_maze))
                        print("nochild: " + str(cur_max_condition))
                        break
                if temp_maze_list.empty():
                    print("localbest: " + str(cur_max_condition))
                    hard_maze_list.put((cur_max_condition, cur_index, cur_maze))
                else:
                    for len_queue in range(0, k):
                        if temp_maze_list.empty():
                            break
                        a = temp_maze_list.get()
                        print(a[0])
                        new_maze_list.put(a)
            for len_queue in range(0, m):
                if new_maze_list.empty():
                    break
                cur_maze_list.put(new_maze_list.get())
        return hard_maze_list.get()

    def Maze_Generate(self, d):
        """
        Algorithm to generate a solvable random maze
        p is chosen as a random number
        """
        p = random.uniform(0.3, 0.4)
        tmp_m = Maze(p, d)
        while not tmp_m.solve('dfs').has_path:
            tmp_m = Maze(p, d)
        tmp_m.env[0][0] = 0
        tmp_m.env[d - 1][d - 1] = 0
        return tmp_m

    def Maze_Score(self, m, alg):
        """
        Algorithm to determine whether a maze is hard
        alg: algorithm we use to solve the maze
        """
        if m.solve(alg) is False:
            return ori_maze.dim
        else:
            if alg == 'dfs':
                params_m = m.solve(alg)
                return params_m.max_fringe_size
            elif alg == 'a*':
                params_m = m.solve(alg)
                return params_m.nodes_expanded
            else:
                print("Only support dfs and a*")

    def Maze_merge(self, m1, m2, d, strategy):
        """
        take in two maze and merge them into one solvable maze
        strategy: strategy we use to merge two maze,
            if set to 1, simply cut them into half and merge
            if set to 0, split each into 4 parts and merge them
        """
        m1_n = deepcopy(m1)
        m2_n = deepcopy(m2)
        if strategy == 1:
            m3 = Maze(1, d)
            m3.env = m1_n.env[0:int(d / 2)] + m2_n.env[int(d / 2):d]
        else:
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
        """
        pick a maze form the list according to the probability given
        """
        x = random.uniform(0, sum(weight))
        cumulative_weight = 0.0
        for m_list, m_weight in zip(m_list, weight):
            cumulative_weight += m_weight
            if x < cumulative_weight:
                break
        return m_list

    def genetic_algorithm(self, alg):
        """
        maze_population: the queue we use when performing bfs
        path(i, j): stores the previous point we visit before step into (i, j)
        visited_node: the point already visited by either side of bfs
        node_expanded: all the point we visited in the algorithm
        node_meet: the point where two bfs meet
        """
        # Generate Parents
        maze_population = []
        for t in range(self.population):
            maze_population.append(deepcopy(self.Maze_Generate(ori_maze.dim)))
        weight = []

        for k in range(self.genetic_iter):
            print("k=", k)
            weight.clear()
            weight = [self.Maze_Score(maze_population[x], alg) for x in range(self.population)]
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
                    weight.append(self.Maze_Score(merge_m, alg))
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
        return maze_population[best_index]


if __name__ == "__main__":
    ori_maze = Maze(0.1, 30)
    sa = LocalSearch(ori_maze)

    max_fringe_size, i, maze1 = sa.beam_search("a*", 10, 30)
    path = maze1.solve("a*").path
    print(max_fringe_size)
    printGraph(maze1, path)


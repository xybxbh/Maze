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
        # parameters for simulated annealing
        self.sa_tem = 1
        self.sa_tmin = 1e-8
        self.sa_delta = 0.98

        # parameters for genetic search
        self.population = 1000
        self.genetic_iter = 20
        self.breed_cnt = 1000
        self.mutant_rate = 0.6

        # parameters for beam search
        self.members = 10
        self.k_best = 30

    # replace the old maze by the new one
    # bound maze update and num_ob update together
    # using for simulated annealing
    def update_maze_param(self, maze):
        self.cur_maze = maze
        self.num_ob = self.get_obstruction_num()

    # count the number of obstacle in the current maze
    # using for simulated annealing
    def get_obstruction_num(self):
        count = 0
        for i in range(self.cur_maze.dim):
            for j in range(self.cur_maze.dim):
                if self.cur_maze.env[i][j] == 1:
                    count += 1
        return count

    # generate new maze that add or remove obstacle based on current maze
    # using this function for simulated annealing
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
                # solvability certification in the sa function
                break
            else:
                print(x, y)
                it += 1
            if it == 1000:
                print('iter protection')
                break
        return new_maze

    # generate new maze that add obstacles based on current maze
    # using this function for beam search
    def gen_maze_bs(self, alg, cur_maze, iter):
        it = 0
        new_maze = deepcopy(cur_maze)
        while(True):
            change = 0
            # add three obstacles at first three iteration
            if iter < 30:
                for i in range(0, 3):
                    x = random.randint(0, cur_maze.dim - 1)
                    y = random.randint(0, cur_maze.dim - 1)
                    if cur_maze.env[x][y] == 0:
                        new_maze.env[x][y] = 1
                        change = 1
            # add one obstacle
            else:
                x = random.randint(0, cur_maze.dim - 1)
                y = random.randint(0, cur_maze.dim - 1)
                if cur_maze.env[x][y] == 0:
                    new_maze.env[x][y] = 1
                    change = 1
            if change == 1:
                if new_maze.solve(alg).has_path == False:
                    it += 1
                else:
                    break
            else:
                it += 1
            if it == 1000:
                print('iter protection')
                return False
        return new_maze

    # compare the fitness function between the new maze and the current maze
    # using this function for simulated annealing
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

    # simulated annealing
    def simulated_annealing(self, alg, condition_param):
        while self.sa_tem > self.sa_tmin:
            new_maze = self.gen_maze(self.cur_maze.solve(alg).path)
            if not new_maze.solve(alg).has_path:
                continue
            p = random.random()
            comparator = self.comparator(alg, condition_param, new_maze)
            # e^700+ will be out of stack
            p_sa = 1 / (1 + math.exp(0 - comparator / self.sa_tem)) if (0 - comparator / self.sa_tem) < 500 else 0
            if comparator > 0 or p < p_sa:
                self.update_maze_param(new_maze)
            self.sa_tem = self.sa_tem * self.sa_delta

    # choose the fitness function property
    # three properties: max_fringe_size, total_nodes_expanded, and total_path_length
    def get_search_condition(self, param_name, solution_param):
        if param_name == "max_fringe_size":
            return solution_param.max_fringe_size
        elif param_name == "total_nodes_expanded":
            return len(solution_param.nodes_expanded)
        elif param_name == "total_path_length":
            return len(solution_param.path)
        return False

    # generate m initial mazes for beam search
    def gen_ini_maze_bs(self, alg, condition_param):
        ini_maze_list = queue.PriorityQueue()
        index = 0
        for i in range(0, self.members):
            maze = Maze(self.ori_maze.occ_rate, self.ori_maze.dim)
            solution_param = maze.solve(alg)
            while solution_param.has_path == False:
                maze = Maze(self.ori_maze.occ_rate, self.ori_maze.dim)
                solution_param = maze.solve(alg)
            ini_maze_list.put((-self.get_search_condition(condition_param, solution_param), index, maze))
            index += 1
        print("ini_finished")
        return (ini_maze_list, index)

    # beam search function
    # m members simultaneously search and each members search k candidates, then from m * k candidates choose m new members
    def beam_search(self, alg, condition_param):
        cur_maze_list, index = self.gen_ini_maze_bs(alg, condition_param)
        hard_maze_list = queue.PriorityQueue()
        new_maze_list = queue.PriorityQueue()
        beam_search_iter = 0
        while not cur_maze_list.empty():
            beam_search_iter += 1
            # maximal iteration times
            if beam_search_iter == 100:
                break
            # each members search k candidates
            for i in range(0, self.members):
                if cur_maze_list.empty():
                    break
                temp_maze_list = queue.PriorityQueue()
                cur_max_condition, cur_index, cur_maze = cur_maze_list.get()
                print("cur_best_size: " + str(cur_max_condition))
                for j in range(0, self.k_best):
                    new_maze = self.gen_maze_bs(alg, cur_maze, beam_search_iter)
                    if new_maze:
                        new_path_param = new_maze.solve(alg)
                        # new maze is better than current maze
                        if self.get_search_condition(condition_param, new_path_param) <= -cur_max_condition:
                            continue
                        temp_maze_list.put((-self.get_search_condition(condition_param, new_path_param), index, new_maze))
                        index += 1
                    # cannot generate solvable maze from current maze
                    # that means current maze is local best
                    else:
                        hard_maze_list.put((cur_max_condition, cur_index, cur_maze))
                        break
                # no better child, also means local best
                if temp_maze_list.empty():
                    hard_maze_list.put((cur_max_condition, cur_index, cur_maze))
                # prepare for next iteration
                else:
                    for len_queue in range(0, self.k_best):
                        if temp_maze_list.empty():
                            break
                        a = temp_maze_list.get()
                        print(a[0])
                        new_maze_list.put(a)
            # choose m new members to search
            for len_queue in range(0, self.members):
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
                return len(params_m.nodes_expanded)
            elif alg == 'bfs':
                params_m = m.solve(alg)
                return len(params_m.path)
            else:
                print("Only support bfs, dfs and a*")

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
    ori_maze = Maze(0.3, 50)
    hm = LocalSearch(ori_maze)

    # beam search
    max_fringe_size, i, maze_bs = hm.beam_search("bfs", "total_path_length")
    solution_param = maze_bs.solve("bfs")
    printGraph(maze_bs, solution_param.path, "bfs", (solution_param.max_fringe_size, solution_param.nodes_expanded))

    # genetic algorithm
    maze_gene = hm.genetic_algorithm('dfs')
    solution_param = maze_gene.solve('dfs')
    printGraph(maze_gene, solution_param.path, "dfs", (solution_param.max_fringe_size, solution_param.nodes_expanded))

    # simulated annealing
    hm.simulated_annealing("bfs", "total_path_length")
    solution_param = hm.cur_maze.solve('bfs')
    printGraph(hm.cur_maze, solution_param.path, "bfs", (solution_param.max_fringe_size, solution_param.nodes_expanded))


from Maze import *
from copy import deepcopy
import random
import numpy as np

class MazeState(Maze):
    def __init__(self, p, d, q):
        Maze.__init__(self, p, d)
        self.cur_pos = (0, 0)
        self.fla_rate = q

        self.ant_num = 1000
        self.iter_time = 200
        self.weight = []
        self.heuristic = []
        self.alpha = 1
        self.beta = 1
        self.rou = 1
        self.path = [(0, 0)]

        self.re_init_check()
        self.env[0][self.dim - 1] = -1  # set fire after reinitialize check

    def re_init_check(self):
        # a better way is to add start and end node in search methods, but deepcopy is more convenient in coding
        self_reverse = deepcopy(self)
        for row in range(self_reverse.dim):
            self_reverse.env[row] = list(reversed(self_reverse.env[row]))
        while self.solve("dfs").has_path is False or self_reverse.solve('dfs').has_path is False:
            print('reinit')
            self.env = [[np.random.binomial(1, self.occ_rate) for col in range(self.dim)] for row in range(self.dim)]
            for row in range(self_reverse.dim):
                self_reverse.env[row] = list(reversed(self.env[row]))

    def update_maze(self):
        p = random.random()
        temp_maze = deepcopy(self)
        for row in range(self.dim):
            for col in range(self.dim):
                node1 = (row, col)
                if self.env[row][col] == 0 and p > temp_maze.hf_survivalrate((row, col)):
                    self.env[row][col] = -1

    def get_valid_neighbors(self, cur_x, cur_y):
        # if useless after calculating suviral rate, just count k here
        res = []
        # valid: in the range, no obstruction, may be on fire
        # used by survival rate calculation and weight solution
        if cur_x + 1 < self.dim and self.env[cur_x + 1][cur_y] != 1:
            res.append((cur_x + 1, cur_y))
        if cur_y + 1 < self.dim and self.env[cur_x][cur_y + 1] != 1:
            res.append((cur_x, cur_y + 1))
        if cur_x - 1 >= 0 and self.env[cur_x - 1][cur_y] != 1:
            res.append((cur_x - 1, cur_y))
        if cur_y - 1 >= 0 and self.env[cur_x][cur_y - 1] != 1:
            res.append((cur_x, cur_y - 1))
        return res

    def hf_choose(self, function, node1, node2):
        if function == "survivalrate":
            w1, w2 = 30, -1
            return self.hf_survivalrate(node1)*w1 + self.hf_manhattan(node1, node2)*w2
        return False

    def hf_survivalrate(self, node1):
        (x, y) = node1
        if self.env[x][y] == -1:
            return 0
        neighbors = self.get_valid_neighbors(x, y)
        k = 0
        for node in neighbors:
            (xx, yy) = node
            if self.env[xx][yy] == -1:
                k += 1
        return pow((1 - self.fla_rate), k)

    def weight_solution(self):
        w1, w2 = 30, -1
        (cur_x, cur_y) = self.cur_pos
        neighbors = self.get_valid_neighbors(cur_x, cur_y)
        weights = {}
        for node in neighbors:
            if node not in self.path:
                weights[node] = self.hf_survivalrate(node)*w1 + self.hf_manhattan(node, (self.dim - 1, self.dim - 1))*w2
        if not weights:
            for node in neighbors:
                weights[node] = self.hf_survivalrate(node)*w1 + self.hf_manhattan(node, (self.dim - 1, self.dim - 1))*w2
        return max(weights, key=weights.get)

    def weight_2step_solution(self):
        (cur_x, cur_y) = self.cur_pos
        neighbors = self.get_valid_neighbors(cur_x, cur_y)
        weights = {}
        for node in neighbors:  # cant be none
            if node not in self.path:   # get_valid_neighbors is used by other methods, so here
                temp = {'sur': self.hf_survivalrate(node), 'dis': self.hf_manhattan(node, (self.dim - 1, self.dim - 1))}
                weights[node] = temp
        if not weights:
            for node in neighbors:
                temp = {'sur': self.hf_survivalrate(node), 'dis': self.hf_manhattan(node, (self.dim - 1, self.dim - 1))}
                weights[node] = temp
        if not weights:
            print('neighbors size', len(neighbors))
        # print(weights)
        max_sur = {x: y for x, y in weights.items() if y['sur'] == weights[max(weights, key=lambda x: weights[x]['sur'])]['sur']}
        min_dis = [x for x, y in max_sur.items() if y['dis'] == max_sur[min(max_sur, key=lambda x: max_sur[x]['dis'])]['dis']]
        if len(min_dis) > 0:    # should be > 0
            return random.sample(min_dis, 1)[0]
        print('size'. len(neighbors), len(weights), len(max_sur), len(min_dis))

    def weight_2step_solution_goal_first(self):
        (cur_x, cur_y) = self.cur_pos
        neighbors = self.get_valid_neighbors(cur_x, cur_y)
        weights = {}
        for node in neighbors:  # cant be none
            if node not in self.path:   # get_valid_neighbors is used by other methods, so here
                temp = {'sur': self.hf_survivalrate(node), 'dis': self.hf_manhattan(node, (self.dim - 1, self.dim - 1))}
                weights[node] = temp
        if not weights:
            for node in neighbors:
                temp = {'sur': self.hf_survivalrate(node), 'dis': self.hf_manhattan(node, (self.dim - 1, self.dim - 1))}
                weights[node] = temp
        if not weights:
            print('neighbors size', len(neighbors))
        # print(weights)

        min_dis = {x: y for x, y in weights.items() if y['dis'] == weights[min(weights, key=lambda x: weights[x]['dis'])]['dis']}
        max_sur = [x for x, y in min_dis.items() if y['sur'] == min_dis[max(min_dis, key=lambda x: min_dis[x]['sur'])]['sur']]
        
        if len(max_sur) > 0:    # should be > 0
            return random.sample(max_sur, 1)[0]
        print('size'. len(neighbors), len(weights), len(max_sur), len(min_dis))

    def update_path(self, alg):
        if alg == 'astar':
            params = self.astarsearch_solution('survivalrate', self.cur_pos)
            if params.has_path:
                self.cur_pos = params.path[1]
                self.path.append(self.cur_pos)
        elif alg == 'weight':
            self.cur_pos = self.weight_solution()
            self.path.append(self.cur_pos)
        elif alg == 'weight_sur_first':
            self.cur_pos = self.weight_2step_solution()
            self.path.append(self.cur_pos)
        elif alg == 'weight_dis_first':
            self.cur_pos = self.weight_2step_solution_goal_first()
            self.path.append(self.cur_pos)
        else:
            index = self.path.index(self.cur_pos)
            if index + 1 >= len(self.path):
                self.cur_pos = (self.dim - 1, 0)
                return
            self.cur_pos = self.path[index + 1]
            return

    def path_choosing(self, path):
        # valid: in the range, no obstruction
        local_weight = []
        (cur_x, cur_y) = self.cur_pos
        for nodes in self.get_valid(cur_x, cur_y, path):
            (x, y) = nodes
            local_weight.append(pow(self.weight[x][y], self.alpha) * pow(self.heuristic[x][y], self.beta))
        x = random.uniform(0, sum(local_weight))
        if sum(local_weight) == 0:
            return False
        cumulative_weight = 0.0
        (xx, yy) = (0, 0)
        for node in self.get_valid(cur_x, cur_y, path):
            (xx, yy) = node
            cumulative_weight += pow(self.weight[xx][yy], self.alpha) * pow(self.heuristic[xx][yy], self.beta)
            if x < cumulative_weight:
                break
        return xx, yy

    def aco(self):
        # initialize
        origin_env = deepcopy(self.env)
        print(origin_env)
        p = self.solve('bfs').path
        self.weight = [[(1 / 4 * self.dim) for col in range(self.dim)] for row in range(self.dim)]
        self.heuristic = [[1 / (self.hf_manhattan((row, col), (self.dim - 1, self.dim - 1)) + 1)for col in range(self.dim)] for row in range(self.dim)]
        for node in p:
            (xx, yy) = node
            self.weight[xx][yy] = 1 / len(p)
        for ii in range(self.iter_time):
            print("iter:", ii)
            tmp_weight = self.weight
            for j in range(self.ant_num):
                self.env = deepcopy(origin_env)
                tmp_path = [(0, 0)]
                self.cur_pos = (0, 0)
                while True:
                    # for kk in range(self.dim):
                    #     print(self.env[kk])
                    # print(self.cur_pos)
                    # print(tmp_path)
                    # print("--------------------")

                    if self.cur_pos == (self.dim - 1, self.dim - 1):
                        # break out
                        status_ = 'win'
                        break
                    if self.path_choosing(tmp_path) is not False:
                        self.cur_pos = self.path_choosing(tmp_path)
                        tmp_path.append(deepcopy(self.cur_pos))
                    else:
                        # no path to take
                        status_ = 'die'
                        break
                    self.update_maze()
                    (cur_x, cur_y) = self.cur_pos
                    if self.env[cur_x][cur_y] == -1:
                        # die
                        status_ = 'die'
                        break
                if status_ == 'win':
                    for t in range(len(tmp_path)):
                        (x, y) = tmp_path[t]
                        tmp_weight[x][y] += 1 / len(tmp_path)
            self.weight = [[tmp_weight[row][col] * self.rou for col in range(self.dim)] for row in range(self.dim)]
        # generate solution
        success = False
        while not success:
            self.env = deepcopy(origin_env)
            solution = [(0, 0)]
            self.cur_pos = (0, 0)
            success = True
            while solution[-1] != (self.dim - 1, self.dim - 1):
                if self.path_choosing(solution) is False:
                    print("No Path")
                    success = False
                    break
                self.cur_pos = self.path_choosing(solution)
                solution.append(deepcopy(self.cur_pos))
            self.cur_pos = (0, 0)
        return solution

    def generate_path(self, alg):
        if alg == 'astar':
            pass
        elif alg == 'weight':
            pass
        elif alg == 'weight_sur_first':
            pass
        elif alg == 'weight_dis_first':
            pass
        elif alg == 'aco':
            self.path = self.aco()
        else:
            self.path = self.solve(alg).path

    def get_fire_num(self): # for test stdout
        count = 0
        for i in range(self.dim):
            for j in range(self.dim):
                if self.env[i][j] == -1:
                    count += 1
        return count


def experiment(maze_state, sol):
    maze_state.cur_pos = (0, 0)
    maze_state.generate_path(sol)
    while True:
        # print('updating')
        maze_state.update_path(sol)
        maze_state.update_maze()
        # print(maze_state.cur_pos)
        (cur_x, cur_y) = maze_state.cur_pos
        if maze_state.env[cur_x][cur_y] == -1:
            return False
        if maze_state.cur_pos == (maze_state.dim - 1, maze_state.dim - 1):
            return True
    return False


    #update

if __name__ == "__main__":
    count = 0
    for i in range(100):
        init_state = MazeState(0.2, 30, 0.5)
        status = experiment(init_state, 'weight_dis_first')
        print(i, status, init_state.cur_pos, init_state.get_fire_num())
        if status:
        # if experiment(init_state):
            count += 1
    print(count/100)

    # init_state = MazeState(0.2, 30, 0.5)
    # print(experiment(init_state, 'weight_2step'))

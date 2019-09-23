from Maze import *
from copy import deepcopy
import random

class MazeState(Maze):
    def __init__(self, p, d, q):
        Maze.__init__(self, p, d)
        self.cur_pos = (0, 0)
        self.fla_rate = q
        self.env[0][self.dim - 1] = -1
        self.flammability = 0.1

        self.ant_num = 1000
        self.iter_time = 1000
        self.weight = []
        self.alpha = 0.1


    def update_maze(self):
        p = random.random()
        for row in range(self.dim):
            for col in range(self.dim):
                node1 = (row, col)
                if p > self.hf_survivalrate((row, col)):
                    self.env[row][col] = -1

    def fire_spreading(self):
        for row in range(self.dim):
            for col in range(self.dim):
                if self.env[row][col] == -1 or self.env[row][col] == 1:
                    break
                flaming_neighbors = 0
                if row + 1 < self.dim and self.env[row + 1][col] == -1:
                    flaming_neighbors += 1
                if col + 1 < self.dim and self.env[row][col + 1] == -1:
                    flaming_neighbors += 1
                if row - 1 < self.dim and self.env[row - 1][col] == -1:
                    flaming_neighbors += 1
                if col - 1 < self.dim and self.env[row][col - 1] == -1:
                    flaming_neighbors += 1
                flaming_rate = 1 - pow(1 - self.flammability, flaming_neighbors)
                if flaming_rate >= random.uniform(0, 1):
                    self.env[row][col] = -1

    def get_valid_neighbors(self, cur_x, cur_y):
        # if useless after calculating suviral rate, just count k here
        res = []
        # valid: in the range, no obstruction
        if cur_x + 1 < self.dim and self.env[cur_x + 1][cur_y] == 0:
            res.append((cur_x + 1, cur_y))
        if cur_y + 1 < self.dim and self.env[cur_x][cur_y + 1] == 0:
            res.append((cur_x, cur_y + 1))
        if cur_x - 1 >= 0 and self.env[cur_x - 1][cur_y] == 0:
            res.append((cur_x - 1, cur_y))
        if cur_y - 1 >= 0 and self.env[cur_x][cur_y - 1] == 0:
            res.append((cur_x, cur_y - 1))
        return res

    def hf_choose(self, function, node1, node2):
        if function == "survivalrate":
            w1, w2 = 0, 0
            return self.hf_survivalrate(node1)*w1 + self.hf_manhattan(node1, node2)*w2
        return False

    def hf_survivalrate(self, node1):
        (x, y) = node1
        neighbors = self.get_valid_neighbors(x, y)
        k = 0
        for node in neighbors:
            (xx, yy) = node
            if self.env[xx][yy] == -1:
                k += 1
        return pow((1 - self.fla_rate), k)

    def update_path(self):
        params = self.astarsearch_solution('survivalrate', self.cur_pos)
        if params.has_path:
            self.cur_pos = params.path[1]
        return

    def path_choosing(self, cur_x, cur_y, path):
        # valid: in the range, no obstruction
        total_weight = 0
        local_weight = [0, 0, 0, 0]
        if cur_x + 1 < self.dim and self.env[cur_x + 1][cur_y] == 0 and (cur_x + 1, cur_y) not in path:
            local_weight[0] = self.weight[cur_x + 1][cur_y]
        if cur_y + 1 < self.dim and self.env[cur_x][cur_y + 1] == 0 and (cur_x, cur_y + 1) not in path:
            local_weight[1] = self.weight[cur_x][cur_y + 1]
        if cur_x - 1 >= 0 and self.env[cur_x - 1][cur_y] == 0 and (cur_x - 1, cur_y) not in path:
            local_weight[2] = self.weight[cur_x - 1][cur_y]
        if cur_y - 1 >= 0 and self.env[cur_x][cur_y - 1] == 0 and (cur_x, cur_y - 1) not in path:
            local_weight[3] = self.weight[cur_x][cur_y - 1]
        x = random.uniform(0, sum(local_weight))
        if sum(local_weight) == 0:
            return False
        total_weight += local_weight[0]
        if total_weight >= x:
            return cur_x + 1, cur_y
        total_weight += local_weight[1]
        if total_weight >= x:
            return cur_x, cur_y + 1
        total_weight += local_weight[2]
        if total_weight >= x:
            return cur_x - 1, cur_y
        total_weight += local_weight[3]
        if total_weight >= x:
            return cur_x, cur_y - 1

    def aco(self):
        # initialize
        origin_env = deepcopy(self.env)
        self.weight = [[(1 / 2 * self.dim) for col in range(self.dim)] for row in range(self.dim)]
        for i in range(self.iter_time):
            tmp_weight = []
            self.env = origin_env
            for j in range(self.ant_num):
                tmp_path = [(0, 0)]
                (cur_x, cur_y) = tmp_path[-1]
                while True:
                    if cur_x == self.dim - 1 and cur_y == self.dim - 1:
                        # break out
                        status = 'win'
                        break
                    if self.path_choosing(cur_x, cur_y, tmp_path) is not False:
                        (cur_x, cur_y) = self.path_choosing(cur_x, cur_y, tmp_path)
                        tmp_path.append(deepcopy((cur_x, cur_y)))
                    else:
                        # no path to take
                        status = 'die'
                        break
                    self.fire_spreading()
                    if self.env[cur_x][cur_y] == -1:
                        # die
                        status = 'die'
                        break
                if status == 'win':
                    for t in range(len(tmp_path)):
                        (x, y) = tmp_path[t]
                        tmp_weight[x][y] += 1 / len(tmp_path)
            self.weight = [[tmp_weight[row][col] * self.alpha for col in range(self.dim)] for row in range(self.dim)]
        # generate solution
        self.env = origin_env
        solution = [(0, 0)]
        (new_x, new_y) = (0, 0)
        while solution[-1] != (self.dim - 1, self.dim - 1):
            (new_x, new_y) = self.path_choosing(new_x, new_y, solution)
            solution.append(deepcopy((new_x, new_y)))
        return solution

def experiment(maze_state):
    while True :
        print('updating')
        maze_state.update_path()
        maze_state.update_maze()
        print(maze_state.cur_pos)
        (cur_x, cur_y) = maze_state.cur_pos
        if maze_state.env[cur_x][cur_y] == -1:
            return False
        if maze_state.cur_pos == (maze_state.dim - 1, maze_state.dim - 1):
            return True
    return False




    #update

if __name__ == "__main__":
    init_state = MazeState(0.2, 30, 0.2)
    experiment(init_state)
    
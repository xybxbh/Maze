from Maze import *
import random

class MazeState(Maze):
    def __init__(self, p, d, q):
        Maze.__init__(self, p, d)
        self.cur_pos = (0, 0)
        self.fla_rate = q
        self.env[0][self.dim - 1] = -1

    def update_maze(self):
        p = random.random()
        for row in range(self.dim):
            for col in range(self.dim):
                node1 = (row, col)
                if p > self.hf_survivalrate((row, col)):
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

def experiment(maze_state):
    while(True):
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

if __name__ == "__main__":
    init_state = MazeState(0.2, 30, 0.2)
    experiment(init_state)
    
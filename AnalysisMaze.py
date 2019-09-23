import numpy as np
from Maze import *
import matplotlib.pyplot as plt

import datetime

class AnalysisMaze(object):

    def __init__(self, p, d):
        self.occ_rate = p
        self.dim = d

    def path_max_fringe_size_dfs(self):
        output = []
        for i in range(0, 1000):
            maze = Maze(self.occ_rate, self.dim)
            path_param = maze.solve("dfs")
            if path_param.has_path:
                output.append((len(path_param.path), path_param.max_fringe_size))
        output.sort()
        print(output)
        return output

    def alg_runningtime(self, alg):
        while True:
            starttime = datetime.datetime.now()
            maze = Maze(self.occ_rate, self.dim)
            solution_param = maze.solve(alg)
            if solution_param.has_path:
                endtime = datetime.datetime.now()
                print((endtime - starttime).seconds)
                break

    def print(self, output):
        x = []
        y = []
        for i in range(0, len(output)):
            x.append(output[i][0])
            y.append(output[i][1])
        plt.plot(x, y, '.')
        plt.show()


def cal_runningtime_list():
    alg = ["dfs", "bfs", "a*", "bdbfs"]
    p = [0, 0.1, 0.2, 0.3]
    dim = [10, 50, 100, 250, 500, 100]
    for i in range(0, len(alg)):
        for j in range(0, len(p)):
            for k in range(0, len(dim)):
                a = AnalysisMaze(p[j], dim[k])
                a.alg_runningtime(alg[i])


if __name__ == "__main__":
    cal_runningtime_list()

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

    def anls_runningtime(self, alg):
        starttime = datetime.datetime.now()
        maze = Maze(self.occ_rate, self.dim)
        path = maze.solve(alg)
        endtime = datetime.datetime.now()
        print (endtime - starttime).seconds

    def print(self, output):
        x = []
        y = []
        for i in range(0, len(output)):
            x.append(output[i][0])
            y.append(output[i][1])
        plt.plot(x, y, '.')
        plt.show()


if __name__ == "__main__":
    a = AnalysisMaze(0.2, 30)
    # a.print(a.path_max_fringe_size_dfs())

import numpy as np
from Maze import *
from VisualizationPath import *

import time

class AnalysisMaze(object):

    def __init__(self, p = 0, d = 0):
        self.occ_rate = p
        self.dim = d

    # def path_max_fringe_size_dfs(self):
    #     output = []
    #     for i in range(0, 1000):
    #         maze = Maze(self.occ_rate, self.dim)
    #         path_param = maze.solve("dfs")
    #         if path_param.has_path:
    #             output.append((len(path_param.path), path_param.max_fringe_size))
    #     output.sort()
    #     print(output)
    #     return output

    def cal_runningtime_single(self, alg):
        self.maze = Maze(self.occ_rate, self.dim)
        while True:
            start = time.time()
            solution_param =  self.get_solution_param(alg)
            if solution_param.has_path:
                end = time.time()
                print(end - start)
                break

    def cal_runningtime_list(self):
        alg = ["dfs", "bfs", "a*EU", "a*MH", "bdbfs"]
        p = [0, 0.1, 0.2, 0.3]
        dim = [10, 50, 100, 250, 500, 1000]
        for i in range(0, len(alg)):
            for j in range(0, len(p)):
                for k in range(0, len(dim)):
                    print("p: " + str(p[j]) + " dim: " + str(dim[k]) + " alg: " + alg[i])
                    self.occ_rate = p[j]
                    self.dim = dim[k]
                    self.cal_runningtime_single(alg[i])

    def get_solution_param(self, alg):
        if alg == "a*EU":
            solution_param = self.maze.solve("a*", "Euclidean")
        elif alg == "a*MH":
            solution_param = self.maze.solve("a*")
        else:
            solution_param = self.maze.solve(alg)
        return solution_param

    # def print(self, output):
    #     x = []
    #     y = []
    #     for i in range(0, len(output)):
    #         x.append(output[i][0])
    #         y.append(output[i][1])
    #     plt.plot(x, y, '.')
    #     plt.show()

    def mazerunner(self, alg_list):
        self.maze = Maze(self.occ_rate, self.dim)
        for i in alg_list:
            solution_param = self.get_solution_param(i)
            if solution_param.has_path:
                printGraph(self.maze, solution_param.path, i, (solution_param.max_fringe_size, solution_param.nodes_expanded))
            else:
                printGraph(self.maze, solution_param.has_path)

    def cal_solvability(self):
        p = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        solvability = []
        for i in range(0, len(p)):
            self.occ_rate = p[i]
            has_path_times = 0
            for j in range(0, 1000):
                self.maze = Maze(self.occ_rate, self.dim)
                solution_param =  self.maze.solve("dfs")
                if solution_param.has_path:
                    has_path_times += 1
            solvability.append(has_path_times / 1000)
        print(solvability)
        printSolvability(p , solvability)

    def cal_aver_path_length(self):
        p = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        aver_path_len = []
        for i in range(0, len(p)):
            self.occ_rate = p[i]
            path_length = 0
            has_path_times = 0
            for j in range(0, 1000):
                self.maze = Maze(self.occ_rate, self.dim)
                solution_param = self.maze.solve("bfs")
                if solution_param.has_path:
                    path_length += len(solution_param.path)
                    has_path_times += 1
            aver_path_len.append(path_length / has_path_times)
        printAverPathLen(p, aver_path_len)

    def compare_solution(self, alg_list):
        self.maze = Maze(self.occ_rate, self.dim)
        for i in alg_list:
            solution_param = self.get_solution_param(i)
            if solution_param.has_path:
                printGraph(self.maze, solution_param.path, i, (solution_param.max_fringe_size, solution_param.nodes_expanded))
            else:
                print("NO SOLUTION")






if __name__ == "__main__":
    # a = AnalysisMaze(0.2, 100)
    # a.mazerunner()
    a = AnalysisMaze(0.4, 100)
    a.mazerunner(["bdbfs"])

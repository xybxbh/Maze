from Maze import *
from VisualizationPath import *

import time

class AnalysisMaze(object):

    def __init__(self, p = 0, d = 0):
        self.occ_rate = p
        self.dim = d

    def cal_runningtime_single(self, alg):
        while True:
            self.maze = Maze(self.occ_rate, self.dim)
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

    def mazerunner(self, alg_list = ["dfs", "bfs", "a*EU", "a*MH", "bdbfs"]):
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
            for j in range(0, 500):
                self.maze = Maze(self.occ_rate, self.dim)
                solution_param =  self.get_solution_param("dfs")
                if solution_param.has_path:
                    has_path_times += 1
            solvability.append(has_path_times / 500)
        printSolvability(p , solvability)

    def cal_solvability_3d(self):
        p = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        dim = [30, 100, 200, 500]
        solvability = []
        p_list = []
        dim_list = []
        for i in range(0, len(p)):
            self.occ_rate = p[i]
            for k in range(0, len(dim)):
                has_path_times = 0
                self.dim = dim[k]
                p_list.append(p[i])
                dim_list.append(dim[k])
                for j in range(0, 100):
                    self.maze = Maze(self.occ_rate, self.dim)
                    solution_param =  self.get_solution_param("dfs")
                    if solution_param.has_path:
                        has_path_times += 1
                solvability.append(has_path_times / 100)
        printSolvability3D(p_list, dim_list, solvability)

    def cal_aver_path_length(self):
        p = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        aver_path_len = []
        for i in range(0, len(p)):
            self.occ_rate = p[i]
            path_length = 0
            has_path_times = 0
            for j in range(0, 1000):
                self.maze = Maze(self.occ_rate, self.dim)
                solution_param = self.get_solution_param("a*MH")
                if solution_param.has_path:
                    path_length += len(solution_param.path)
                    has_path_times += 1
            aver_path_len.append(path_length / has_path_times)
        printAverPathLen(p, aver_path_len)

    def compare_astar(self):
        dim = [50, 100, 250, 500, 1000]
        nodes_diff_list = []
        fringe_diff_list = []
        for i in dim:
            self.dim = i
            has_path_times = 0
            nodes_diff = 0
            fringe_diff = 0
            for k in range(0, 100):
                self.maze = Maze(self.occ_rate, self.dim)
                solution_param_MH = self.get_solution_param("a*MH")
                if solution_param_MH.has_path:
                    solution_param_EU = self.get_solution_param("a*EU")
                    nodes_diff += len(solution_param_EU.nodes_expanded) - len(solution_param_MH.nodes_expanded)
                    fringe_diff += len(solution_param_EU.nodes_expanded) - len(solution_param_MH.nodes_expanded)
                    has_path_times += 1
            nodes_diff_list.append(nodes_diff / has_path_times)
            fringe_diff_list.append(fringe_diff / has_path_times)


if __name__ == "__main__":
    a = AnalysisMaze(0.2, 100)
    a.mazerunner(['bdbfs','a*MH'])

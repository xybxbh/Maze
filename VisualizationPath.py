import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

import numpy as np

def printGraph(maze, path, extratext = False, extraparam = (0, [])):
    print(path)
    plt.xlim(0, maze.dim)
    plt.ylim(0, maze.dim)
    plt.xticks(range(0, maze.dim, 1))
    plt.yticks(range(0, maze.dim, 1))
    plt.grid(True, linestyle = "-", color = "black", linewidth = "0.5")
    axes = plt.gca()
    axes.tick_params(left = False, bottom = False, labelbottom = False, labelleft = False)

    data = np.array(maze.env)
    rows, cols = data.shape

    axeslist_x = []
    axeslist_y = []


    if extratext:
        max_fringe_size, nodes_expaned = extraparam
        for i in range(0, maze.dim):
            for j in range(0, maze.dim):
                if data[i][j] == 1:
                    data[i][j] = 2
        for i in range(0, len(nodes_expaned)):
            x, y = nodes_expaned[i]
            data[x][y] = 1
        string = "max fringe size: " + str(max_fringe_size) + "\npath length: " + str(len(path))
        plt.title(extratext)
        plt.text(0, maze.dim * (-0.15) + 2, string)

    # print(maze.env)

    if path == False:
        plt.title("NO SOLUTION")
    else:
        for t in path:
            (y, x) = t
            axeslist_x.append(x + 0.5)
            axeslist_y.append(maze.dim - y - 0.5)
        plt.plot(axeslist_x, axeslist_y, c="red")
    plt.imshow(data, cmap='gray_r', extent=[0, cols, 0, rows])

    plt.show()

def printSolvability(xlist, ylist):
    plt.title("solvability")
    axes = plt.gca()
    for i in range(0, len(ylist)):
        ylist[i] = ylist[i] * 100
    axes.yaxis.set_major_formatter(PercentFormatter())
    plt.plot(xlist, ylist)
    plt.show()

def printAverPathLen(xlist, ylist):
    plt.title("Average Path Length")
    plt.plot(xlist, ylist)
    plt.show()

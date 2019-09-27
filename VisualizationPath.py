import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        string = "max fringe size: " + str(max_fringe_size) + "\npath length: " + str(len(path)) + "\ntotal expanded nodes: " + str(len(nodes_expaned))
        plt.title(extratext + " p = " + str(maze.occ_rate) + " dim = " + str(maze.dim))
        plt.text(0, maze.dim * (-0.14) + 2, string)

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
    plt.grid(axis='y', which='major')
    plt.xlabel("density")
    plt.ylabel("solvability")
    plt.title("density vs solvability")
    axes = plt.gca()
    for i in range(0, len(ylist)):
        ylist[i] = int(ylist[i] * 100)
    for i in range(0, len(ylist)):
        plt.text(xlist[i], ylist[i] + 1, str(ylist[i]) + "%", ha = 'center')
    plt.bar(xlist, ylist, 0.02, color="orange")
    x_ticks_range = np.arange(xlist[0], xlist[len(xlist) - 1] + 0.05, 0.05)
    plt.xticks(x_ticks_range)
    axes.yaxis.set_major_formatter(PercentFormatter())
    plt.plot(xlist, ylist)
    plt.show()

def printSolvability3D(xlist, ylist, zlist):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0, len(zlist)):
        zlist[i] = zlist[i] * 100
    print(zlist)
    ax.plot_trisurf(xlist, ylist, zlist, cmap='rainbow')
    ax.set_zlabel("solvability")
    ax.set_xlabel("density")
    ax.set_ylabel("dimension")
    ax.set_title("density vs dimension vs solvability")
    ax.set_zlim(0, 100)
    ax.zaxis.set_major_formatter(PercentFormatter())

    plt.show()


def printAverPathLen(xlist, ylist):
    plt.grid()
    plt.xlabel("Density")
    plt.ylabel("Average Path Length")
    plt.title("Density vs Average Path Length")
    plt.plot(xlist, ylist, 'b*-')
    plt.show()

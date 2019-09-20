import matplotlib.pyplot as plt
import numpy as np

def printGraph(maze, path):
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

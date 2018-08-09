from heapq import heappop, heappush
pathvalue = 120

def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def find_path_astar(maze,s,e):
    start, goal = s, e
    pr_queue = []
    heappush(pr_queue, (0 + heuristic(start, goal), 0, "", start))
    visited = set()
    graph = maze2graph(maze)
    while pr_queue:
        _, cost, path, current = heappop(pr_queue)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(pr_queue, (cost + heuristic(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "NO WAY!"

def maze2graph(maze):
    height = len(maze)
    width = len(maze[0]) if height else 0
    graph = {(i, j): [] for j in range(width) for i in range(height) if maze[i][j] <pathvalue}
    for row, col in graph.keys():
        if row < height - 1 and maze[row + 1][col] < pathvalue:
            graph[(row, col)].append(("S", (row + 1, col)))
            graph[(row + 1, col)].append(("N", (row, col)))
        if col < width - 1 and  maze[row][col + 1]<pathvalue:
            graph[(row, col)].append(("E", (row, col + 1)))
            graph[(row, col + 1)].append(("W", (row, col)))
    return graph

def markPath(maze, path,start):
    maze[start[0]][start[1]] = pathvalue
    cur = start
    for direction in path:
        if direction == 'E':
            maze[cur[0]][cur[1]+1] = pathvalue
            cur = (cur[0],cur[1]+1)
        if direction == 'W':
            maze[cur[0]][cur[1]-1] = pathvalue
            cur = (cur[0], cur[1] - 1)
        if direction == 'N':
            maze[cur[0]-1][cur[1]] = pathvalue
            cur = (cur[0]-1, cur[1])
        if direction == 'S':
            maze[cur[0]+1][cur[1]] = pathvalue
            cur = (cur[0]+1, cur[1])
    return maze
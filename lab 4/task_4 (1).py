import heapq
from collections import deque

start = "A"
goal = "G"

graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": [],
    "G": [],
}

wgraph = {
    "A": [("B", 2), ("C", 1)],
    "B": [("A", 2), ("D", 4), ("E", 6)],
    "C": [("F", 1)],
    "D": [("G", 1)],
    "E": [("G", 8)],
    "F": [("D", 1)],
    "G": [],
}

def bfs(g):
    queue = deque([[start]])
    visited = [start]
    order = []
    while queue:
        path = queue.popleft()
        node = path[-1]
        order.append(node)
        if node == goal:
            return order, path
        for n in g[node]:
            if n not in visited:
                visited.append(n)
                queue.append(path + [n])
    return order, None

def dfs():
    stack = [[start]]
    visited = []
    order = []
    while stack:
        path = stack.pop()
        node = path[-1]
        if node in visited:
            continue
        visited.append(node)
        order.append(node)
        if node == goal:
            return order, path
        for n in reversed(graph[node]):
            if n not in visited:
                stack.append(path + [n])
    return order, None

def dls(node, target, limit, path, depth):
    if node == target:
        return "FOUND", path
    if depth == limit:
        return "CUTOFF", None
    cutoff = False
    for n in graph[node]:
        if n in path:
            continue
        result, p = dls(n, target, limit, path + [n], depth + 1)
        if result == "FOUND":
            return result, p
        if result == "CUTOFF":
            cutoff = True
    if cutoff:
        return "CUTOFF", None
    return "FAILURE", None

def iddfs(max_depth):
    tried = []
    result = "FAILURE"
    for limit in range(1, max_depth + 1):
        tried.append(limit)
        result, path = dls(start, goal, limit, [start], 0)
        if result == "FOUND":
            return result, path, tried
    return result, None, tried

def ucs():
    queue = [(0, start, [start])]
    visited = []
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return cost, path
        for n, w in wgraph[node]:
            if n not in visited:
                heapq.heappush(queue, (cost + w, n, path + [n]))
    return None, None

def path_cost(path):
    total = 0
    for i in range(len(path) - 1):
        for n, w in wgraph[path[i]]:
            if n == path[i + 1]:
                total += w
    return total

order, path = bfs(graph)
print("BFS order:", order)
print("BFS path:", path)

order, path = dfs()
print("DFS order:", order)
print("DFS path:", path)

for limit in [1, 2, 3, 4]:
    result, path = dls(start, goal, limit, [start], 0)
    print("DLS limit", limit, ":", result, path)

for m in [1, 2, 3, 4]:
    result, path, tried = iddfs(m)
    print("IDDFS max depth", m, ":", result, path, "limits tried:", tried)

print("Missing goal:", dls(start, "H", 4, [start], 0)[0])

cost, path = ucs()
print("UCS path:", path)
print("UCS cost:", cost)
print("Sum of edge weights:", path_cost(path))

order, path = bfs({k: [n for n, w in v] for k, v in wgraph.items()})
print("BFS on weighted graph:", path, "cost:", path_cost(path))

"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""
from collections import deque


class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path
    """

    def shortestPath(self, grid, source, destination):
        # write your code here
        assert grid and source and destination
        if source.x == destination.x and source.y == destination.y:
            return 0

        m, n = len(grid), len(grid[0])
        dx = [1, 1, -1, -1, 2, 2, -2, -2]
        dy = [2, -2, 2, -2, 1, -1, 1, -1]
        visited = [[False for _ in range(n)] for _ in range(m)]
        visited[source.x][source.y] = True
        steps = 0
        q = deque()
        q.append(source)

        while q:
            size = len(q)
            steps += 1
            for i in range(size):
                cur = q.popleft()
                for j in range(8):
                    nei = Point(cur.x + dx[j], cur.y + dy[j])
                    if self.isValidPos(nei, grid, visited):
                        if nei.x == destination.x and nei.y == destination.y:
                            return steps
                        visited[nei.x][nei.y] = True
                        q.append(nei)
        return -1

    def isValidPos(self, p, grid, visited):
        return p.x >= 0 and p.y >= 0 and p.x < len(grid) and p.y < len(grid[0]) and not visited[p.x][p.y] and not \
        grid[p.x][p.y]
























from collections import deque


class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """

    def numIslands(self, grid):
        # write your code here
        if not grid:
            return 0
        m = len(grid)
        n = len(grid[0])
        counter = 0
        visited = [[False for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] and not visited[i][j]:
                    counter += 1
                    self.markIsland((i, j), grid, visited)
        return counter

    # mark island at pos as visited
    def markIsland(self, pos, grid, visited):
        # pos: (x, y)
        if not grid:
            return
        assert grid[pos[0]][pos[1]]

        q = deque()
        q.append(pos)
        visited[pos[0]][pos[1]] = True

        dx = [0, 0, -1, 1]
        dy = [-1, 1, 0, 0]

        while q:
            cur = q.popleft()
            for i in range(4):
                nei = (cur[0] + dx[i], cur[1] + dy[i])
                if self.isInboundAndUnvisited(nei, grid, visited):
                    visited[nei[0]][nei[1]] = True
                    q.append(nei)

    def isInboundAndUnvisited(self, pos, grid, visited):
        x = pos[0]
        y = pos[1]
        return x >= 0 and y >= 0 and x < len(grid) and y < len(grid[0]) and not visited[x][y]


test = [[1,0], [0,1]]

s = Solution()

c = s.numIslands(test)

print(c)






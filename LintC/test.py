class Solution:
    """
    @param n: The number of queens.
    @return: The total number of distinct solutions.
    """

    def totalNQueens(self, n):
        # write your code here
        assert n >= 0, "Invalid input of n"
        if n < 2:
            return n
        occupied_rows = set()
        board = [[False for _ in range(n)] for _ in range(n)]
        self.count = 0
        self._dfs(0, board, set())
        return self.count

    def _dfs(self, start, board, occupied_rows):
        if start == len(board):
            self.count += 1
            return
        for i in range(start, len(board)):
            for j in range(0, len(board)):
                if not self._isValid(i, j, board, occupied_rows):
                    continue
                occupied_rows.add(j)
                board[i][j] = True
                self._dfs(start + 1, board, occupied_rows)
                occupied_rows.remove(j)
                board[i][j] = False

    def _isValid(self, a, b, board, occupied_rows):
        m = len(board)
        n = len(board[0])
        # if out of boundary
        if a < 0 or a >= m or b < 0 or b >= n:
            return False
        # check rows
        if b in occupied_rows:
            return False

        c = [a, b]
        delta_1 = [-1, -1]
        delta_2 = [-1, 1]

        # check delta_1 direction:
        while c[0] >= 0 and c[1] >= 0:
            c[0] += delta_1[0]
            c[1] += delta_1[1]
            if board[c[0]][c[1]]:
                return False

        # check delta_2 direction:
        c = [a, b]
        while c[0] >= 0 and c[1] < n:
            c[0] += delta_2[0]
            c[1] += delta_2[1]
            if board[c[0]][c[1]]:
                return False
        return True

s = Solution()
res = s.totalNQueens(5)

print(res)

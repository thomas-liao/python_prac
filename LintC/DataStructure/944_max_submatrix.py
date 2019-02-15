class Solution:
    """
    @param matrix: the given matrix
    @return: the largest possible sum
    """

    def maxSubmatrix(self, matrix):
        # write your code here
        assert matrix is not None and len(matrix) > 0 and len(matrix[0]) > 0, "Invalid input of matrix."
        # pad matrix
        n = len(matrix)
        table = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                table[i][j] = matrix[i - 1][j - 1]

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                table[i][j] = table[i][j - 1] + table[i - 1][j] - table[i - 1][j - 1] + table[i][j]

        record = None
        record_idx = [-1 for _ in range(4)]
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for k in range(1, n + 1):
                    for l in range(1, n + 1):
                        window_sum = table[k][l] - table[i][j] + table[i][l] + table[k][j]
                        if window_sum > record:
                            record = window_sum
                            record_idx = [i, j, k, l]
        return 5




test = [[1,3,-1],[2,3,-2],[-1,-2,-3]]

s = Solution()
res = s.maxSubmatrix(test)

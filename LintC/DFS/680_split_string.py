# 10-31-18 practice again
class Solution:
    """
    @param: : a string to be split
    @return: all possible split string array
    """

    def splitString(self, s):
        if s is None:
            return []

        if len(s) == 0:
            return [[]]
        element = list(s)

        result = []
        self._dfs_backtracking(element, 0, [], result)
        return result


    def _dfs_backtracking(self, element, start, subset, result):
        if start == len(element):
            result.append(subset.copy())
            return
        if start > len(element):
            return  # overshooting return

        for i in range(1, 3):
            subset.append("".join(element[start:start + i]))
            self._dfs_backtracking(element, start + i, subset, result)
            subset.pop()


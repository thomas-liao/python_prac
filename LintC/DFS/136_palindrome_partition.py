# 10-31-18
class Solution:
    """
    @param: s: A string
    @return: A list of lists of string
    """

    def partition(self, s):
        if s is None:
            return []
        if len(s) == 0:
            return [""]
        result = []
        self._dfs(s, 0, [], result)
        return result

    def _dfs(self, s, start, subset, result):
        if start == len(s):
            result.append(subset.copy())
            return
        if start > len(s):
            return

        for i in range(start, len(s)):
            if self.isPalindrome(s[start:i + 1]):
                subset.append(s[start:i + 1])
                self._dfs(s, i + 1, subset, result)
                subset.pop()
        return

    def isPalindrome(self, s):
        if s is None or len(s) == 0:
            return True
        l = 0
        r = len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True


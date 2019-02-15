
# 1 O3 algorithm
# solution 1. O(n^3) with slightly optimization
class Solution:
    """
    @param s: input string
    @return: the longest palindromic substring
    """

    def longestPalindrome(self, s):
        # write your code here
        if s is None:
            return ""
        if len(s) < 2:
            return s
        for i in range(len(s), 0, -1):
            for j in range(len(s) - i + 1):
                if self.isPalindrome(s[j:j + i]):
                    return s[j:j + i]
        return ""

    def isPalindrome(self, s):
        if len(s) < 2:
            return True
        l = 0
        r = len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

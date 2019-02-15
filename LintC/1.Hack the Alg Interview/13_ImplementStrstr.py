class Solution:
    """
    @param source:
    @param target:
    @return: return the index
    """

    def strStr(self, source, target):
        # Write your code here
        # corner cases
        if source is None or target is None or len(target) > len(source):
            return -1
        for i in range(len(source) - len(target) + 1):
            if self._isMatch(source, target, i):
                return i
        return -1

    def _isMatch(self, source, target, idx):
        for i in range(len(target)):
            if target[i] != source[idx + i]:
                return False
        return True
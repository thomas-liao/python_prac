class Solution:
    """
    @param A: A an integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """

    def totalOccurrence(self, A, target):
        # write your code here
        if not A:
            return 0
        A.sort()

        first_idx = self.firstIdx(A, target)
        last_idx = self.lastIdx(A, target)
        if first_idx == -1:
            return 0
        else:
            return last_idx - first_idx + 1

    def firstIdx(self, A, target):
        left = 0
        right = len(target) - 1

        while left + 1 < right:
            mid = left + (right - left) // 2
            if A[mid] >= target:
                right = mid
            else:
                left = mid

        if A[left] == target:
            return left
        if A[right] == target:
            return right
        return -1

    def lastIdx(self, A, target):
        left = 0
        right = len(target) - 1

        while left + 1 < right:
            mid = left + (right - left) // 2
            if A[mid] <= target:
                left = mid
        else:
            right = mid

        if A[right] == target:
            return right
        if A[left] == target:
            return left
        return -1

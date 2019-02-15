# Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
# You don't need to care the order of combinations, but you should make sure the numbers in a combination are sorted.
# Have you met this question in a real interview?  Yes
# Problem Correction
# Example
# Given n = 4 and k = 2, a solution is:




# # class Solution:
# #     """
# #     @param n: Given the range of numbers
# #     @param k: Given the numbers of combinations
# #     @return: All the combinations of k numbers out of 1..n
# #     """
# #     def combine(self, n, k):
#         # write your code here
#         if k > n:
#             k = n
#         nums = [i for i in range(1, n + 1)]
#         subset = []
#         result = []
#         self._dfs(nums, 0, [], result, k)
#         return result

#     def _dfs(self, nums, start, subset, result, k):
#         if len(subset) > k:
#             return

#         if len(subset) == k:
#             result.append(subset.copy())
#             return

#         for i in range(start, len(nums)):
#             subset.append(nums[i])
#             self._dfs(nums, i + 1, subset, result, k)
#             subset.pop()


class Solution:
    """
    @param n: Given the range of numbers
    @param k: Given the numbers of combinations
    @return: All the combinations of k numbers out of 1..n
    """

    def combine(self, n, k):
        assert n > 0 and k > 0 and k <= n

        elements = [i for i in range(1, n + 1)]
        result = []
        self.dfs(elements, 0, [], result, k, 0)
        return result

    def dfs(self, elements, start, subset, result, k, counter):
        if start >= len(elements) or counter + len(elements) - start < k:  # not enough elements to continue
            return
        for i in range(start, len(elements)):
            subset.append(elements[i])
            counter += 1
            if counter == k:
                result.append(subset.copy())
            self.dfs(elements, i + 1, subset, result, k, counter)
            counter -= 1
            subset.pop()





# combination sum I, each element can be used unlimited times but all elements are positive
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """

    def combinationSum(self, candidates, target):
        # if candidates is None or not len(candidates) or not len(candidates[0]):
        #     return []
        if candidates is None:
            return []

        candidates = sorted(list(set(candidates)))
        assert candidates[0] > 0
        result = []
        self._dfs(candidates, 0, [], result, 0, target)
        return result

    def _dfs(self, candidates, start, subset, result, sum_, target):
        if start >= len(candidates) or sum_ > target:
            return  # over
        for i in range(start, len(candidates)):
            sum_ += candidates[i]
            subset.append(candidates[i])
            if sum_ == target:
                result.append(subset.copy())
            # self._dfs(candidates, start, subset, result, sum_, target) # bug
            self._dfs(candidates, i, subset, result, sum_, target)
            sum_ -= candidates[i]
            subset.pop()
















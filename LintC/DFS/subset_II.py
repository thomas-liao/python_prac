class Solution:
    """
    @param nums: A set of numbers.
    @return: A list of lists. All valid subsets.
    """
    def subsetsWithDup(self, nums):
        # write your code here
        if nums is None:
            return []
        if len(nums) == 0:
            return [[]]
        nums = sorted(nums)
        result = []
        self._dfs(nums, 0, [], result)
        return result

    def _dfs(self, nums, start, subset, result):
        # if start >= len(nums): # this is wrong... corner case must be though twice
        #     return
        if start > len(nums):
            return
        result.append(subset.copy())

        for i in range(start, len(nums)):
            # deduplication
            if i != 0 and nums[i] == nums[i-1] and i != start:
                continue
            subset.append(nums[i])
            self._dfs(nums, i + 1, subset, result)
            subset.pop()


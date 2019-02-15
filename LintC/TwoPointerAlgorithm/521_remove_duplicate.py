# Solution 1,  O(nlogn) time without extra space.
class Solution:
    """
    @param: nums: an array of integers
    @return: the number of unique integers
    """
    def deduplication(self, nums):
        if nums is None or len(nums) == 0:
            return 0
        if len(nums) < 2:
            return 1
        idx = 1
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                nums[idx] = nums[i]
                idx += 1 # ready for next writing on this location
        return idx



# Solution 2, O(n),e.g. use set
class Solution:
    """
    @param: nums: an array of integers
    @return: the number of unique integers
    """

    def deduplication(self, nums):
        if nums is None or len(nums) == 0:
            return 0
        if len(nums) < 2:
            return 1
        visited = set()
        visited.add(nums[0])
        idx = 1
        for i in range(1, len(nums)):
            if nums[i] not in visited:
                visited.add(nums[i])
                nums[idx] = nums[i]
                idx += 1
        return idx

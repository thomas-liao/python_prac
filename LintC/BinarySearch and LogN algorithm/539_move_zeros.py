# 前向 快慢双指针， 快指针每次往前挪动1， 慢指针用来录入（swap) 每次快指针指向的元素非0的值

class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """

    def moveZeroes(self, nums):
        if nums is None or len(nums) < 2:
            return

        slow = fast = 0

        while fast < len(nums):
            if nums[fast] != 0:
                # swap slow and fast value to persistant non-zero value at fast pointe
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
            fast += 1


class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """

    def maxSubArray(self, nums):
        # write your code here
        assert nums is not None and len(nums) > 0, "Invalid input of nums"
        start = -1
        end = 0
        pre_min = 0
        sum_ = 0
        record = None

        for i in range(len(nums)):
            sum_ += nums[i]
            val = sum_ - pre_min
            if record is None or val > record:
                record = val
                end = i

            if sum_ < pre_min:
                pre_min = sum_
                start = i

        # (start, end]
        # return nums[start+1:end+1]
        return record




















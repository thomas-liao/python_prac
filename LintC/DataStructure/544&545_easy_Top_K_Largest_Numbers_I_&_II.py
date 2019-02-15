# Description
# Given an integer array, find the top k largest numbers in it.
# Have you met this question in a real interview?  Yes
# Example
# Given [3,10,1000,-99,4,100] and k = 3.
# Return [1000, 100, 10].

# 8-30
import heapq


# python heap by default is min heap i guess
class Solution:
    """
    @param nums: an integer array
    @param k: An integer
    @return: the top k largest numbers in array
    """

    def topk(self, nums, k):
        # write your code here
        if nums is None or len(nums) == 0:
            return None
        self.pool = nums
        heapq.heapify(self.pool)
        while len(self.pool) > k:
            heapq.heappop(self.pool)

        return sorted(self.pool, reverse=True)


##
# 545
import heapq


class Solution:
    """
    @param: k: An integer
    """

    def __init__(self, k):
        # do intialization if necessary
        self.k = k
        self.pool = []

    """
    @param: num: Number to be added
    @return: nothing
    """

    def add(self, num):
        # write your code here
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, num)
            return
        # heapq.heapify(self.pool) redundant..?
        if self.pool[0] < num:
            heapq.heapreplace(self.pool, num)
            # this one is equal to following 2 lines
            # heapq.heappop(self.pool)
            # heapq.heappush(self.nums, num)
        return

    """
    @return: Top k element
    """

    def topk(self):
        # write your code here
        ret = self.pool.copy()
        return sorted(ret, reverse=True)


"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = bDefinition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""


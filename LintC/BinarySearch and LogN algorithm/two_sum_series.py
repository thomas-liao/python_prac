# 56, most basic two sum question O(n) speed and O(n) time
# or if need to be sorted, can have O(n) space and O(logn) time

# 注意，返回index跟返回value的做法是不一样的，返回index你没法sort然后用double pointer的方法来搞。

class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2)
    """

    def twoSum(self, numbers, target):
        # write your code here
        if numbers is None or len(numbers) < 2:
            return [-1, -1]

        dict_ = {}
        for i in range(len(numbers)):
            if numbers[i] in dict_.keys():
                return [dict_[numbers[i]], i]
            else:
                dict_[target - numbers[i]] = i
        return [-1, -1]


# 607 two sum III data structure design
class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """

    def __init__(self):
        self.nums = []

    def add(self, number):
        # write your code here
        self.nums.append(number)

    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """

    def find(self, value):
        # write your code here
        dict_ = {}
        for i in range(len(self.nums)):
            a = self.nums[i]
            if a in dict_.keys():
                return True
            else:
                dict_[value - a] = i
        return False

# 608 two sum two
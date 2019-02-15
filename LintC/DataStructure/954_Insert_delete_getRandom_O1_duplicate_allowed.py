import random


class RandomizedCollection(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums, self.pos = [], {}

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.pos.keys():
            self.pos[val] = set()
        self.nums.append(val)
        self.pos[val].add(len(self.nums) - 1)

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.pos.keys():
            return  # nothing to remove
        set_ = self.pos[val]
        idx = random.choice(list(set_))
        last_idx = len(self.nums) - 1
        last_val = self.nums[last_idx]
        self.nums[last_idx], self.nums[idx] = self.nums[idx], self.nums[last_idx]
        self.nums.pop()  # remove element
        self.pos[last_val].remove(last_idx)
        self.pos[last_val].add(idx)
        self.pos[val].remove(idx)
        # check corner case - if pos[val] is empty
        if not self.pos[val]:
            del self.pos[val]

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        return random.choice(self.nums)

# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
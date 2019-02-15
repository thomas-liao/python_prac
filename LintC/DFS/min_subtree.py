"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


# 10-30-18
class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """

    def findSubtree(self, root):
        # write your code here
        self.res = None
        self.record = None
        self.dfs(root)
        return self.res

    def dfs(self, root):
        if root is None:
            return 0
        l = self.dfs(root.left)
        r = self.dfs(root.right)
        combined = l + r + root.val
        if self.record is None or combined < self.record:
            self.record = combined
            self.res = root
        return combined




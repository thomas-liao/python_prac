# """
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []
# """
#
# from collections import deque
#
#
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []
#
# class Solution:
#     """
#     @param: node: A undirected graph node
#     @return: A undirected graph node
#     """
#
#     def cloneGraph(self, node):
#         # write your code here
#         if node is None:
#             return None
#
#         record = {}
#
#         # traverse the graph
#         q = deque()
#         visited = set()
#         q.append(node)
#         visited.add(node)
#
#         while q:
#             cur = q.popleft()
#             # map original graph nodes to new graph node
#             record[cur] = UndirectedGraphNode(cur.label)
#             for nei in cur.neighbors:
#                 if nei not in visited:
#                     visited.add(nei)
#                     q.append(nei)
#
#         # add related connections
#         for n in list(visited):
#             for nei in n.neighbors:
#                 record[n].neighbors.append(record[nei])
#
#         return record[node]
#
#
#

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
        return








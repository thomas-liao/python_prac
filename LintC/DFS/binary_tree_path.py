# # 10-30
#
# class Solution:
#     """
#     @param root: the root of the binary tree
#     @return: all root-to-leaf paths
#     """
#
#     def binaryTreePaths(self, root):
#         # write your code here
#         if root is None:
#             return []
#         result = []
#         self._backtracking(root, [], result)
#         ret = []
#         for res in result:
#             ret.append("->".join(map(str, res)))
#         return ret
#
#     # return [[]], each[] contains value from root to leave
#     def _backtracking(self, root, subset, result):
#         if root is None:
#             return
#         # root leave
#         if root.left is None and root.right is None:
#             subset.append(root.val)
#             result.append(subset.copy())
#             subset.pop()
#             return
#
#         if root.left:
#             subset.append(root.val)
#             self._backtracking(root.left, subset, result)
#             subset.pop()
#
#         if root.right:
#             subset.append(root.val)
#             self._backtracking(root.right, subset, result)
#             subset.pop()
#         return


a = [1,2,3,4,5]

b = map(str, a)

print(list(b))

# import random
#
# class Solution:
#     """
#     @param A: an integer array
#     @return: nothing
#     """
#
#     def sortIntegers2(self, A):
#         # write your code here
#         if not A:
#             return A
#
#         self.quickSort(A, 0, len(A) - 1)
#         return A
#
#     def quickSort(self, A, left, right):
#         if left >= right:
#             return
#         idx = self.partition(A, left, right, random.choice(range(left, right + 1)))
#         self.quickSort(A, left, idx - 1)
#         self.quickSort(A, idx + 1, right)
#         return
#
#     def partition(self, A, left, right, p_idx):
#         if left >= right:
#             return right
#         p_val = A[p_idx]
#         A[p_idx], A[right] = A[right], A[p_idx]
#
#         idx = left
#         for i in range(left, right):
#             if A[i] < p_val:
#                 A[i], A[idx] = A[idx], A[i]
#                 idx += 1
#
#         A[idx], A[right] = A[right], A[idx]
#         return idx
#
# class Solution:
#     """
#     @param A: an integer array
#     @return: nothing
#     """
#     def sortIntegers2(self, A):
#         self.mergeSort(A, 0, len(A) - 1)
#         return A
#
#     def mergeSort(self, A, left, right):
#         if left >= right:
#             return A
#         if left == right - 1:
#             if A[left] > A[right]:
#                 A[left], A[right] = A[right], A[left]
#             return
#         partition = (left + right) // 2
#         self.mergeSort(A, left, partition)
#         self.mergeSort(A, partition + 1, right)
#         self.merge(A, left, right, partition)
#         return
#
#     def merge(self, A, left, right, partition):
#         if left >= right:
#             return
#         temp = [0 for _ in range(right-left+1)]
#         A_left = A[left:partition + 1]
#         A_right = A[partition + 1:right+1]
#
#         idx = l_idx = r_idx = 0
#         while l_idx < len(A_left) and r_idx < len(A_right):
#             if A_left[l_idx] < A_right[r_idx]:
#                 temp[idx] = A_left[l_idx]
#                 l_idx += 1
#             else:
#                 temp[idx] = A_right[r_idx]
#                 r_idx += 1
#             idx += 1
#
#         if l_idx < len(A_left):
#             # temp.extend(A_left[l_idx:len(A_left)]) # bug
#             temp[idx:] = A_left[l_idx:len(A_left)]
#         else:
#             temp[idx:] = A_right[r_idx:len(A_right)]
#
#         A[left:right + 1] = temp
#         return
#
# s = Solution()
# test = [i for i in range(1000)]
# s.sortIntegers2(test)
#
#
#
# print(test)
#
#
# from collections import deque
#
# q = deque()
#
#
#
#
#
#
#



from collections import deque


from collections import deque













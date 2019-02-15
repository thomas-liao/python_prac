# ord(s)

# naive solution
# class Solution:
#     """
#     @param str: str: the given string
#     @return: char: the first unique character in a given string
#     """
#     def firstUniqChar(self, str):
#         # Write your code here
#         if str is None or len(str) == 0:
#             return None
#         char_count = [0 for _ in range(256)]
#         for s in str:
#             char_count[ord(s)] += 1
#         for s in str:
#             if char_count[ord(s)] == 1:
#                 return s
#         return None
#

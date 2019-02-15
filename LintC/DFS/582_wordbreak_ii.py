# class Solution:
#     """
#     @param: s: A string
#     @param: wordDict: A set of words.
#     @return: All possible sentences.
#     """
#
#     def wordBreak(self, s, wordDict):
#         # write your code here
#         if s is None:
#             return []
#         if len(s) == 0:
#             return [""]
#         result = []
#         self.dfs(s, wordDict, {}, result, len(s))
#         return result
#
#     def dfs(self, s, wordDict, tempDict, result, n):
#         if len(s) == 0:
#             return ['']
#         if s in tempDict:
#             return tempDict[s]
#
#         for i in range(len(s)):
#             if s[:i + 1] in wordDict:
#                 front = s[:i + 1]
#                 if s[i + 1:] in tempDict:
#                     back = tempDict[s[i + 1:]]
#                 else:
#                     back = self.dfs(s[i + 1:], wordDict, tempDict, result, n)
#                     if back is None:
#                         return None
#                 combined = []
#                 for b in back:
#                     c = front + " " + b
#                     combined.append(c)
#                 if len(s) == n:
#                     result.append(combined)
#
#         # if no combination of words available on s
#         return combined
#
#
# s = Solution()
# res = s.wordBreak("lintcode", ["de","ding","co","code","lint"])
# print(res)''


class Solution:
    """
    @param: :  A list of integers
    @return: A list of unique permutations
    """

    def permuteUnique(self, nums):
        if nums is None:
            return []
        if len(nums) == 0:
            return [[]]

        nums.sort()
        result = []

        self._permuteUniqueHelper(nums, set(), [], result)
        return result

    def _permuteUniqueHelper(self, nums, visited, subset, result):
        if len(subset) == len(nums):
            result.append(subset.copy())
            return

        # failure proof
        if len(subset) > len(nums):
            return


        for i in range(len(nums)):
            if i in visited:
                continue
            visited.add(i)
            subset.append(nums[i])
            self._permuteUniqueHelper(nums, visited, subset, result)
            visited.remove(i)
            subset.pop()


class Solution:
    """
    @param: s: A string
    @param: wordDict: A set of words.
    @return: All possible sentences.
    """
    def wordBreak(self, s, wordDict):
        # write your code here
        if s is None:
            return []
        if len(s) == 0:
            return [""]
        result = []
        self.dfs(s, wordDict, {}, result, len(s))
        return result

    def dfs(self, s, wordDict, tempDict, result, n):
        if len(s) == 0:
            return [""]
        for i in range(len(s)):
            if s in tempDict:
                return tempDict[s]

            if s[:i+1] in wordDict:
                front = s[:i+1]
                if s[i+1:] in tempDict:
                    back = tempDict[s[i+1:]]
                else:
                    back = self.dfs(s[i+1:])
                    if back is None:
                        return None
                combined = []
                for b in back:
                    c = front + " "+ b
                    combined.append(c)
                if len(s) == n:
                    result.append(combined)

        # if no combination of words available on s
        return None

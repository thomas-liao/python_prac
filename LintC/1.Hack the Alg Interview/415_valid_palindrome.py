class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """

    def isPalindrome(self, s):
        # write your code here
        if s is None:
            return False
        if len(s) == 0:
            return True

        extract_str = []
        for c in s:
            if c.isalnum():
                extract_str.append(c)

        extract_str = "".join(extract_str)
        extract_str = extract_str.lower()
        l_idx = 0
        r_idx = len(extract_str) - 1

        while l_idx < r_idx:
            if extract_str[l_idx] != extract_str[r_idx]:
                return False
            l_idx += 1
            r_idx -= 1
        return True


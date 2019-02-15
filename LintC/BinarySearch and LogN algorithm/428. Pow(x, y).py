
class Solution:
    """
    @param x: the base number
    @param n: the power number
    @return: the result
    """

    def myPow(self, x, n):
        # write your code here
        # corner cases
        assert not (x == 0 and n == 0), "0^0 is undefined"
        if x == 0:
            return 0
        if x == 1 or n == 0:
            return 1

        # corner case: x < 0
        neg_flag = True if x < 0 else False
        if x < 0:
            x = - x

        # corner case: if n < 0
        if n < 0:
            x = 1 / x
            n = -n
        save_res = {}
        base_res = self.rec(x, n, save_res)
        if n % 2 == 1 and neg_flag:
            base_res *= -1
        return base_res

    def rec(self, x, n, save_res):
        if n == 0:
            return 1
        if n == 1:
            return x
        if n in save_res.keys():
            return save_res[n]
        res = self.rec(x, n // 2, save_res) * self.rec(x, n - n // 2, save_res)

        save_res[n] = res
        return res
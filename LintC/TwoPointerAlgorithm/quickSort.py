class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """

    def sortColors2(self, colors, k):
        if colors is None or len(colors) < 2:
            return

        self.quickSort(colors, 0, len(colors) - 1)
        return

    def quickSort(self, colors, left, right):
        if left < right:
            pi = self.partition(colors, left, right, right)  # using right index for partition
            self.quickSort(colors, left, pi - 1)
            self.quickSort(colors, pi + 1, right)

    def partition(self, colors, left, right, p_idx):
        if left >= right:
            return right

        # swap p_idx with right index
        colors[p_idx], colors[right] = colors[right], colors[p_idx]

        idx = left
        p_val = colors[right]

        for i in range(left, right):
            if colors[i] < p_val:
                colors[i], colors[idx] = colors[idx], colors[i]
                idx += 1

        # swap pivot to its idx value
        colors[idx], colors[right] = colors[right], colors[idx]
        return idx
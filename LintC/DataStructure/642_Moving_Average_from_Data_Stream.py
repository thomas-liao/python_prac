# Description
# Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.
# Have you met this question in a real interview?  Yes
# Example
# MovingAverage m = new MovingAverage(3);
# m.next(1) = 1 // return 1.00000
# m.next(10) = (1 + 10) / 2 // return 5.50000
# m.next(3) = (1 + 10 + 3) / 3 // return 4.66667
# m.next(5) = (10 + 3 + 5) / 3 // return 6.00000

# (1) maintain a sum , return sum / q.size(), (2) when q.size == size, q.get.... q.put..., update sum (-get, + val)
# api of Queue (from queue import Queue : get(), put(), qsize(), empty())
# 8-28
class MovingAverage:
    """
    @param: size: An integer
    """

    def __init__(self, size):
        # do intialization if necessary
        from queue import Queue
        self.size = size
        self.sum_ = 0.0
        self.q = Queue()

    """
    @param: val: An integer
    @return:  
    """

    def next(self, val):
        # write your code here
        self.sum_ += val
        if self.q.qsize() == self.size:
            self.sum_ -= self.q.get()
        self.q.put(val)
        return self.sum_ / self.q.qsize()

# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param = obj.next(val)



# or
class MovingAverage:
    """
    @param: size: An integer
    """
    def __init__(self, size):
        # do intialization if necessary
        from queue import Queue
        self.size = size
        self.q = Queue()
        self.sum_ = 0.0

    """
    @param: val: An integer
    @return:  
    """
    def next(self, val):
        # write your code here
        self.sum_ += val
        self.q.put(val)
        if self.q.qsize() == self.size + 1:
            self.sum_ -= self.q.get()
        return self.sum_ / self.q.qsize()
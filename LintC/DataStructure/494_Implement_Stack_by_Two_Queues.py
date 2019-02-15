# Description
# Implement a stack by two queues. The queue is first in first out (FIFO). That means you can not directly pop the last element in a queue.
# Have you met this question in a real interview?  Yes
# Example
# push(1)
# pop()
# push(2)
# isEmpty() // return false
# top() // return 2
# pop()
# isEmpty() // return true



# 8-28  hardcore Queue solution... i actually don't like it, i like deque
# deque: appendleft, append, extend, extendleft, index(x, start[,stop]], insert(i, x), pop(), popleft(), remove(value), reverse(),
from queue import Queue


class Stack:
    """
    @param: x: An integer
    @return: nothing
    """

    def __init__(self, ):
        self.q1 = Queue()
        self.q2 = Queue()

    def push(self, x):
        # write your code here
        self.q1.put(x)

    """
    @return: nothing
    """

    def pop(self):
        # write your code here
        if self.q1.empty():
            return  # nothing to pop
        while self.q1.qsize() != 1:
            self.q2.put(self.q1.get())
        self.q1, self.q2 = self.q2, self.q1
        return self.q2.get()

    """
    @return: An integer
    """

    def top(self):
        # write your code here
        while self.q1.qsize() > 1:
            self.q2.put(self.q1.get())
        item = self.q1.get()
        self.q2.put(item)
        self.q1, self.q2 = self.q2, self.q1
        return item

    """
    @return: True if the stack is empty
    """

    def isEmpty(self):
        # write your code here
        return self.q1.empty()




# deque solution.. slightly different API
from collections import deque


class Stack:
    """
    @param: x: An integer
    @return: nothing
    """

    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        # write your code here
        self.q1.append(x)

    """
    @return: nothing
    """

    def pop(self):
        # write your code here
        # while self.q1.size() > 1: #@ no size attribute
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        item = self.q1.popleft()
        self.q1, self.q2 = self.q2, self.q1
        return item

    """
    @return: An integer
    """

    def top(self):
        # write your code here
        # while self.q1.size() > 1: # no size attribute
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        item = self.q1[0]
        self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
        return item

    """
    @return: True if the stack is empty
    """

    def isEmpty(self):
        # write your code here
        # return self.q1.isEmpty() #@ no .isEmpty()
        return len(self.q1) == 0


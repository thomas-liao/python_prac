# when s2 is not empty - just use as queue...

class MyQueue:

    def __init__(self):
        # do intialization if necessary
        self.s1 = []
        self.s2 = []

    """
    @param: element: An integer
    @return: nothing
    """

    def push(self, element):
        # write your code here
        self.s1.append(element)

    """
    @return: An integer
    """

    def pop(self):
        # write your code here
        if self.s2:
            return self.s2.pop()
        while self.s1:
            self.s2.append(self.s1.pop())
        if self.s2:
            return self.s2.pop()
        return None

    """
    @return: An integer
    """

    def top(self):
        # write your code here
        if self.s2:
            return self.s2[-1]
        while self.s1:
            self.s2.append(self.s1.pop())

        if self.s2:
            return self.s2[-1]
        return None
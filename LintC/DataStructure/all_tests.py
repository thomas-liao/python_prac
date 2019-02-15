# # from Queue import Queue
# #
# # q = Queue()
# # q.put(5)
# #
# # print(q[0])
#
#
#
# import time
# from collections import deque
#
# a = [i for i in range(1000000)]
#
# start = time.time()
#
# while a:
#     a.pop(0)
#
# end = time.time()
#
# print('result using list - pop(0): ', (end - start))
#
# aa = [i for i in range(1000000)]
# q = deque()
# q.extend(aa)
#
# start = time.time()
# while q:
#     q.popleft()
# end = time.time()
# print('result using deque - popleft(): ', (end - start))
#
#
#
#
# Here is the summary for in:

# list - Average: O(n)
# set/dict - Average: O(1), Worst: O(n)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param = obj.insert(val)
# param = obj.remove(val)
# param = obj.getRandom()
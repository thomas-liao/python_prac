#
#
# # print(any([True, True, True, True, False]))
#
# # oct 30
# from collections import deque
#
#
# class Solution:
#     """
#     @param org: a permutation of the integers from 1 to n
#     @param seqs: a list of sequences
#     @return: true if it can be reconstructed only one or false
#     """
#
#     def sequenceReconstruction(self, org, seqs):
#         # write your code here
#         flag1 = org is None or len(org) == 0
#         flag2 = seqs is None or len(seqs) == 0 or len(seqs[0]) == 0
#         if flag1 and flag2:
#             return True
#         if flag1 or flag2:
#             return False
#
#         indegree = [0 for _ in range(len(org))]
#         map_ = [set() for _ in range(len(org))]
#
#         for seq in seqs:
#             if seq is None or len(seq) < 2:
#                 continue
#             for i in range(len(seq) - 1):
#                 j = i + 1
#                 u = seq[i] - 1
#                 v = seq[j] - 1
#                 indegree[v] += 1
#                 map_[u].add(v)
#
#         q = deque()
#         for i in range(len(indegree)):
#             if indegree[i] == 0:
#                 q.append(i)
#
#         res = []
#         while q:
#             if len(q) != 1:
#                 return False
#             cur = q.popleft()
#             res.append(cur + 1)
#             for c in map_[cur]:
#                 indegree[c] -= 1
#                 if indegree[c] == 0:
#                     q.append(c)
#
#         return res == org
#
#


from collections import deque

# print(any([True, True, True, True, False]))

# oct 30
from collections import deque


class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """

    def sequenceReconstruction(self, org, seqs):
        # write your code here
        flag1 = org is None or len(org) == 0
        flag2 = seqs is None or len(seqs) == 0 or len(seqs[0]) == 0
        if flag1 and flag2:
            return True
        if flag1 or flag2:
            return False

        indegree = [0 for _ in range(len(org))]
        map_ = [set() for _ in range(len(org))]

        # censor all elements in seq

        for seq in seqs:
            if seq is not None:
                for e in seq:
                    if e not in org:
                        return False

            if seq is None or len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                j = i + 1
                u = seq[i] - 1
                v = seq[j] - 1
                return False
                if v not in map_[u]:  # disgusting corner case - repeated adding indegree
                    map_[u].add(v)
                    indegree[v] += 1

        q = deque()
        for i in range(len(indegree)):
            if indegree[i] == 0:
                q.append(i)

        res = []
        while q:
            if len(q) != 1:
                return False
            cur = q.popleft()
            res.append(cur + 1)
            for c in map_[cur]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    q.append(c)

        return res == org








































from collections import deque


class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """

    def canFinish(self, numCourses, prerequisites):
        # write your code here
        if numCourses == 0 or not prerequisites:
            return True

        # record indegree and dependencies
        indegree = [0 for _ in range(numCourses)]
        map_ = [[] for _ in range(numCourses)]

        for p in prerequisites:
            u = p[0]
            v = p[1]
            indegree[u] += 1
            map_[v].append(u)

        # search for starting courses
        q = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                q.append(i)

        counter = 0
        while q:
            cur = q.popleft()
            counter += 1
            for c in map_[cur]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    q.append(c)
        return counter == numCourses


from collections import deque





# course schedule 2
from collections import deque


class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """

    def findOrder(self, numCourses, prerequisites):
        # write your code here
        assert numCourses >= 0
        if numCourses == 0 or not prerequisites:
            return [i for i in range(numCourses)]

        indegree = [0 for _ in range(numCourses)]
        map_ = [[] for _ in range(numCourses)]

        for p in prerequisites:
            u = p[0]
            v = p[1]
            indegree[u] += 1
            map_[v].append(u)

        q = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                q.append(i)

        final_res = []
        while q:
            cur = q.popleft()
            final_res.append(cur)

            for c in map_[cur]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    q.append(c)

        if len(final_res) == numCourses:
            return final_res

        return []
















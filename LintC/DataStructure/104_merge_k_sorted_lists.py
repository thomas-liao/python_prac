"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
import heapq


class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """

    def mergeKLists(self, lists):
        # write your code here
        if lists is None or len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]

        heap = []
        for ln in lists:
            if ln:
                self.heapPushNode(heap, ln)
        ret = dummy = ListNode(-1)

        while heap:
            curNode = heapq.heappop(heap)[1]  # pop node
            dummy.next = curNode
            curNode = curNode.next
            dummy = dummy.next
            if curNode:
                self.heapPushNode(heap, curNode)  # push the head of the remaining list into the heap

        return ret.next

    def heapPushNode(self, heap, node):
        heapq.heappush(heap, [node.val, node])



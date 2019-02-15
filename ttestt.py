# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

global c

def solution(A):
    # write your code in Python 3.6
    if A is None:
        return 0
    if len(A) <= 2:
        return len(A)
    global c
    c = 0
    helper(A, 0, False, None)
    return c


def helper(A, pos, flag_, last_height):
    # flag_: have we cut one tree
    if pos >= len(A) and flag_:
        global c
        c += 1
        return

    if not flag_:
        if last_height is None or pos < len(A) and A[pos] >= last_height:


            helper(A, pos + 2, True, A[pos])

            helper(A, pos + 1, False, A[pos])
        else:
            return
    if flag_:
        # we have already cut the tree
        if last_height is None or pos < len(A) and A[pos] >= last_height:
            helper(A, pos + 1, True, A[pos])
        else:
            return

res = solution([i for i in range(996)])
print(res)
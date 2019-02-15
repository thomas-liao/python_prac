import multiprocessing as mp
import time



# # you will see if you do not use a lock in multicore.. you wll get repetition value
# def job(v, num):
#     for _ in range(10):
#         time.sleep(0.1)
#         v.value += num
#         print(v.value)
# def multicore():
#     v = mp.Value('i', 0)
#     p1 = mp.Process(target=job, args=(v, 1))
#     p2 = mp.Process(target=job, args=(v, 3))
#     p1.start()
#     p2.start()
#     p1.join()
#     p2.join()


def job(v, num, l):
    l.acquire()
    for _ in range(10):
        time.sleep(0.1)
        v.value += num
        print(v.value)
    l.release()

def multicore():
    l = mp.Lock()
    v = mp.Value('i', 0)
    p1 = mp.Process(target=job, args=(v, 1, l))
    p2 = mp.Process(target=job, args=(v, 3, l))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

# as expected, good
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 13
# 16
# 19
# 22
# 25
# 28
# 31
# 34
# 37
# 40




if __name__ == '__main__':
    multicore()


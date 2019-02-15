import multiprocessing as mp
import time
# if you use pool, your job function can have return..

def job(x):
    return x*x

def multicore():
    pool = mp.Pool(processes=2)
    res = pool.map(job, range(10))
    res = pool.apply_async(job, (2,))
    print(res.get())

    multi_res = [pool.apply_async(job, (i,)) for i in range(20)]

    print([res.get() for res in multi_res])


if __name__ == '__main__':
    multicore()
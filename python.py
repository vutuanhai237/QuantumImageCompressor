import time
import numpy as np
import concurrent.futures

def square(x):
    x = 0
    for i in range(10000000):
        x += 1
    return x**2
if __name__ == '__main__':
    start_time = time.time()
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(square,  range(1, 100))

    # results = [square(x) for x in range(1, 100)]  #46s, 56s

    print(time.time() - start_time)

    # print(results)

    # print(list(results))
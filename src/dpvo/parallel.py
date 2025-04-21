import multiprocessing as mp
import traceback
from multiprocessing import Queue


def pgenerator(func, *args, size=10, **kwargs):
    queue = Queue(maxsize=size)
    process = mp.Process(
        target=_worker, args=(queue, func, args, size, kwargs), daemon=True
    )
    process.start()

    while True:
        try:
            data = queue.get()
            if data is None:
                break
            yield data
        except Exception as e:
            print("Error in parallel generator:", e)
            traceback.print_exc()
            break

    process.join()


def _worker(queue, func, args, size, kwargs):
    try:
        for data in func(*args, **kwargs):
            queue.put(data)
    except Exception as e:
        print("Error in worker:", e)
        traceback.print_exc()
    finally:
        queue.put(None)

import signal, time
from contextlib import contextmanager
from time_limit import time_limit, TimeoutException

def long_function_call():
    while True:
        if time.time() % 1 == 0:
            print('*')

try:
    with time_limit(3):
        long_function_call()
except TimeoutException as msg:
    print(msg)


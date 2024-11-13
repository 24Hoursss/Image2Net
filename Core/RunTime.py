import time
from functools import wraps


# 用于计算函数运行时间，自定义修饰器，带有ms/s/min/hour/day单位
def cal_time(arg=None):
    default_unit = 's'
    units = ['ms', 's', 'min', 'hour', 'day']
    times = [1000, 1, 1 / 60, 1 / 3600, 1 / 3600 / 24]

    if isinstance(arg, str):
        if arg.lower() not in units:
            raise TypeError(f"Invalid time unit: {arg}")
        unit = arg.lower()
    else:
        unit = default_unit

    time_process = times[units.index(unit)]

    def decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            begin_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = (time.perf_counter() - begin_time) * time_process
            print(f"{func.__name__} 用时 {elapsed_time:.6f} {unit}")
            return result

        return wrap

    if callable(arg):
        return decorator(arg)
    else:
        return decorator

def f_range(start, stop, step):
    sum_val = 0
    values = []
    while sum_val < stop:
        values.append(sum_val)
        sum_val += step
    values.append(stop)
    print values
    return values

f_range(0, 10, 2.2)

from std.collections import Set
from std import math
comptime fname = 'day3/input_large'

def largest(array: StringSlice, start: Int, end: Int) raises -> Tuple[Int, Int]:
    so_far = 0
    index = -1
    for i in range(start, end):
        if Int(array[byte=i]) > so_far:
            so_far = Int(array[byte=i])
            index = i
    return so_far, index


comptime stages = 12

def main() raises:
    total = 0
    with open(fname, 'r') as f:
        raw_data = f.read()
        for line in raw_data.split('\n'):
            if line:
                number = 0
                next_start = 0
                for i in range(stages):
                    cur, next_start = largest(line, next_start, len(line) - (stages - i) + 1)
                    next_start += 1
                    number *= 10
                    number += cur
                print(number)
                total += number
    print(total)

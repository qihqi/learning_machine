from std.collections import Set
from std import math
comptime fname = 'day2/input_large'

def num_digits(num: Int) -> Int:
    return Int(math.log10(Float64(num))) + 1


def ranges_of_same_digits(start: Int, end: Int) -> List[Tuple[Int, Int]]:
    result: List[Tuple[Int, Int]] = []


    start_digits = num_digits(start)
    end_digits = num_digits(end)

    current = start

    while start_digits < end_digits:
        new_end = pow(10, start_digits) - 1
        result.append((current, new_end))
        current = new_end + 1
        start_digits += 1

    result.append((current, end))
    return result^


def all_invalid(start: Int, end: Int) -> Int:
    res = 0
    for i in range(start, end + 1):
        str_i = String(i)
        if len(str_i) % 2 == 0:
            half = len(str_i) // 2
            first_part = str_i[byte=0:half]
            second_part = str_i[byte=half:]
            if first_part == second_part:
                res += i
    return res



def multiplier(digits: Int, repeat_length: Int) -> Int:
    start = 1
    mult = pow(10, repeat_length)

    for _ in range(repeat_length, digits, repeat_length):
        start *= mult
        start += 1
    return start

def ceildiv(a: Int, b: Int) -> Int:
    if a % b == 0:
        return a // b
    return a // b + 1

def floordiv(a: Int, b: Int) -> Int:
    return a // b



def find_sum(start: Int, end: Int) -> Int:
    # start and end has same num digits
    digits = num_digits(start)

    var known_adders = Set[Int]()
    result = 0
    for repeat_length in reversed(range(1, digits)):
        if digits % repeat_length == 0:
            print('here ', repeat_length)
            mult = multiplier(digits, repeat_length)
            if mult > 1:
                A = ceildiv(start, mult)
                B = floordiv(end, mult)
                print('here ', start, end, mult, A, B)

                for i in range(A, B + 1):
                    candidate = i * mult
                    if candidate not in known_adders:
                        known_adders.add(i * mult)
                        result += i * mult

    return result


def main() raises:
    all_ranges:List[Tuple[Int, Int]] = []

    with open(fname, 'r') as f:
        raw_data = f.read()
        for ranges in raw_data.split(','):
            if '-' in ranges:
                res = ranges.split('-')
                a = Int(res[0])
                b = Int(res[1])
                all_ranges.extend(ranges_of_same_digits(a, b))

        total = 0
        for a, b in all_ranges:
            print('{}-{} s sum is {}'.format(a, b, find_sum(a, b)))
            total += find_sum(a, b)
        print('TOTAL', total)

def main2() raises:
    print(find_sum(222220, 222224))

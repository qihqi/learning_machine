comptime fname = 'day5/input_large'

struct BitVector:
    var _vec: List[UInt64]
    var _max_size: UInt64

    def __init__(out self, size: Int):
        vector_size = size // 8 + 1

        self._vec: List[UInt64] = []
        self._vec.reserve(Int(vector_size))
        for _ in range(vector_size):
            self._vec.append(UInt64(0))
        self._max_size = size

    def set(mut self, pos: UInt64, value: Bool):
        assert pos < self._max_size
        elem = pos // 8
        bit = pos % 8
        assert UInt64(elem) < len(self._vec[elem]), " element too large"
        if value:
            self._vec[elem] = self._vec[elem] | (1 << bit)
        else:
            self._vec[elem] = self._vec[elem] & (~(1 << bit))

    def get(mut self, pos: UInt64) -> Bool:
        assert pos < self._max_size
        elem = pos // 8
        bit = pos % 8
        return (self._vec[elem] >> bit) & 1 == 1


def total_fresh_from_ranges(ranges: List[Tuple[Int, Int]]) -> Int:
    s = List[Tuple[Int, Bool]]()
    q = List[Tuple[Int, Bool]]()

    for start, end in ranges:
        s.append((start, False))
        s.append((end, True))

    fn cmp_fn(a: Tuple[Int, Bool], b: Tuple[Int, Bool]) capturing -> Bool:
        return a[0] < b[0]

    sort[cmp_fn](Span(s))

    count = 0
    for val, is_end in s:
        if is_end:
            top = q.pop()
            if len(q) == 0:
                count += (val - top[0] + 1)
        else:
            q.append((val, is_end))
    return count




def total_fresh(ranges: List[Tuple[Int, Int]], data: List[Int]):
    total  = 0
    for d in data:
        for start, end in ranges:
            if start <= d <= end:
                total += 1
                break
    print(total)

def main() raises:
    ranges = List[Tuple[Int, Int]]()
    data = List[Int]()
    minimum = 1 << 60
    maximum = 0
    with open(fname, 'r') as f:
        raw_data = f.read()
        for line in raw_data.split('\n'):
            if not line:
                continue
            if '-' in line:
                r = line.split('-')
                ranges.append((Int(r[0]), Int(r[1])))
                print('ranges', Int(r[1]) - Int(r[0]))
                minimum = min(minimum, Int(r[0]))
                maximum = max(maximum, Int(r[0]))
            else:
                data.append(Int(line))
                minimum = min(minimum, Int(line))
                maximum = max(maximum, Int(line))
    print(total_fresh_from_ranges(ranges))

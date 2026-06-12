from std.collections import Set

comptime fname = 'day7/input_large'

def _new_times(size: Int, out times: List[Int]):
    times = []
    for _ in range(size):
        times.append(0)

struct Beam:
    var map: List[String]
    var cur_row: Int
    var columns: Set[Int]
    var splits: Int
    var times: List[Int]


    def __init__(out self, var map: List[String]):
        self.map = map^
        self.cur_row = 0
        self.columns = {}
        self.splits = 0

        self.times = _new_times(len(self.map[0]))
        for i in range(len(self.map[0])):
            if self.map[0][byte=i] == 'S':
                self.columns.add(i)
                self.times[i] = 1



    def move_down(mut self):
        self.cur_row += 1
        new_add: List[Int] = []
        new_remove: List[Int] = []

        new_times = _new_times(len(self.map[0]))
        for t in self.times:
            print(t, end=' ')
        print()

        for c in self.columns:
            assert c < len(self.map[self.cur_row])
            if self.map[self.cur_row][byte=c] == '^':
                if c - 1 >= 0:
                    new_add.append(c - 1)
                    new_times[c - 1] += self.times[c]
                if c + 1 < len(self.map[self.cur_row]):
                    new_add.append(c + 1)
                    new_times[c + 1] += self.times[c]
                new_remove.append(c)
                self.splits += 1
            else:
                new_times[c] += self.times[c]
        for c in new_add:
            self.columns.add(c)
        for c in new_remove:
            try:
                self.columns.remove(c)
            except:
                pass
        self.times = new_times^

    def total_times(self) -> Int:
        total = 0
        for t in self.times:
            total += t
        return total


def main() raises:
    map: List[String] = []
    with open(fname, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.strip():
                map.append(String(line))

    beam: Beam = Beam(map^)
    while beam.cur_row < len(beam.map) - 1:
        beam.move_down()
    print(beam.splits)
    print(beam.total_times())

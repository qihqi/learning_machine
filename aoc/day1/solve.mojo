struct Dial:
    var num: Int
    var tally: Int
    var _started_at_0: Bool

    def __init__(out self, num: Int):
        self.num = num
        self.tally = 0
        self._started_at_0 = (self.num == 0)

    def _tally(mut self):
        times = abs(self.num / 100)
        if self.num <= 0:
            if not self._started_at_0:
                times += 1
        print('Final num is', self.num)
        print('times num is', times)
        self.tally += times
        self.num %= 100
        self._started_at_0 = (self.num == 0)

    def step(mut self, steps: Int):
        self.num += steps
        self._tally()


comptime example = """L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
"""


def main() raises:
    lock = Dial(50)

    with open('day1/input.txt', 'r') as f:
        content = f.read()
        # content = String(example)
        content = content.replace('L', '-').replace('R', '')
        for line in content.split('\n'):
            try:
                lock.step(Int(line))
            except:
                pass
    print(lock.tally)

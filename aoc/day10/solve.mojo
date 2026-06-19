comptime fname = 'day10/input_large'
from std.collections import Deque
from std.collections import Set

import heap

struct State(heap.OrderedCollectionElement):
    var location: List[Int]
    var value: Int
    var dist: Int

    def __init__(out self, var location: List[Int], var value: Int, var dist: Int):
        self.location = location^
        self.value = value
        self.dist = dist

    def __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    def __gt__(self, other: Self) -> Bool:
        return self.value > other.value

    def print(self):
        print('(', end='')
        for i in self.location:
            print(i, end=', ')
        print(')', self.value, self.dist)

def _dist(f: List[Int], to: List[Int]) -> Int:
    max_dist = 1 << 50
    current = 0
    for i, j in zip(f, to):
        if i > j:
            return max_dist
        current = max(j - i, current)
    return current


struct Line:
    var buttons: List[List[Int]]
    var goal: List[Bool]
    var jolt_goal: List[Int]

    def __init__(out self, var goal: List[Bool], var jolt_goal: List[Int], var buttons: List[List[Int]]):
        self.goal = goal^
        self.buttons = buttons^


        def cmp(a: List[Int], b: List[Int]) capturing -> Bool:
            return len(a) > len(b)

        sort[cmp](Span(self.buttons))

        self.jolt_goal = jolt_goal^

    def print(self):
        print('[', end='')
        for i in range(len(self.goal)):
            if self.goal[i]:
                print('#', end='')
            else:
                print('.',end='')
        print(']', end=' ')

        for button in self.buttons:
            print('(', end='')
            for number in button:
                print(number, end=',')
            print(')', end=' ')
        print()


    def get_neighbors(self, current: List[Bool]) -> List[List[Bool]]:
        neighbors: List[List[Bool]] = []

        for b in self.buttons:
            cur_copy = current.copy()
            for num in b:
                cur_copy[num] = not cur_copy[num]
            neighbors.append(cur_copy^)
        return neighbors^

    def get_neighbors_jolt(self, current: List[Int]) -> List[List[Int]]:
        neighbors: List[List[Int]] = []

        for b in self.buttons:
            cur_copy = current.copy()
            for num in b:
                cur_copy[num] += 1
            neighbors.append(cur_copy^)
        return neighbors^


    def min_clicks_jolt2(self) raises -> Int:
        start: List[Int] = []
        for _ in range(len(self.goal)):
            start.append(0)

        g_score: Dict[List[Int], Int] = {}
        f_score: Dict[List[Int], Int] = {}

        g_score[start.copy()] = 0
        f_score[start.copy()] = _dist(start, self.jolt_goal)

        open_set: List[State] = []
        open_set.append(State(start^, f_score[start], 0))
        var visited = Set[List[Int]]()

        while open_set:
            var current_o = heap.heappop(open_set)
            var current = current_o.unsafe_take()

            if current.location in visited:
                continue
            visited.add(current.location)

            if _dist(current.location, self.jolt_goal) == 0:
                print('found')
                current.print()
                return current.dist

            neighbors = self.get_neighbors_jolt(current.location)

            for n in neighbors:
                tent_g_score = current.dist + 1
                if (n not in g_score) or (tent_g_score < g_score[n]):
                    g_score[n.copy()] = tent_g_score
                    f_score[n.copy()] = tent_g_score + _dist(n, self.jolt_goal)
                nstate = State(n.copy(), f_score[n], current.dist + 1)
                heap.heappush(open_set, nstate)

        return -1



    def min_clicks_jolt(self) raises -> Int:
        state: List[Int] = []
        for _ in range(len(self.goal)):
            state.append(0)

        var queue = Deque[Tuple[List[Int], Int]]()
        var visited = Set[List[Int]]()

        res = -1

        def eq(a: List[Int], b: List[Int]) -> Bool:
            for i, j in zip(a, b):
                if j != i:
                    return False
            return True

        def anylarger(a: List[Int], b: List[Int]) -> Bool:
            for i, j in zip(a, b):
                if i > j:
                    return True
            return False

        queue.appendleft((state^, 0))

        while queue:
            current = queue.pop()

            if current[0] in visited:
                continue
            visited.add(current[0])

            if anylarger(current[0], self.jolt_goal):
                continue

            if eq(current[0], self.jolt_goal):
                res = current[1]
                break


            neighbors = self.get_neighbors_jolt(current[0])
            for n in neighbors:
                queue.append((n.copy(), current[1]+ 1))

        return res


    def min_clicks(self) raises -> Int:
        state: List[Bool] = []
        for _ in range(len(self.goal)):
            state.append(False)

        var queue = Deque[Tuple[List[Bool], Int]]()
        var visited = Set[List[Bool]]()

        res = -1

        def eq(a: List[Bool], b: List[Bool]) -> Bool:
            for i, j in zip(a, b):
                if j != i:
                    return False
            return True

        queue.appendleft((state^, 0))

        while queue:
            current = queue.popleft()

#            for i in range(len(current[0])):
#                if current[0][i]:
#                    print('#', end='')
#                else:
#                    print('.',end='')

            if current[0] in visited:
                continue
            visited.add(current[0])
            if eq(current[0], self.goal):
                res = current[1]
                break

            neighbors = self.get_neighbors(current[0])
            for n in neighbors:
                queue.append((n.copy(), current[1]+ 1))

        return res





def main() raises:

    total = 0
    with open(fname, 'r') as f:
        content = f.read()

        for lines in content.split('\n'):
            items = lines.split()
            if not items:
                continue

            # parse goals
            goal: List[Bool] = []
            for i in range(1, len(items[0]) - 1):
                goal.append(items[0][byte=i] == '#')

            # parse buttons:
            buttons: List[List[Int]] = []
            for b in items[1:-1]:
                buttons.append([])
                if b.startswith('('):
                    numbers = b[byte=1:-1].split(',')
                    for n in numbers:
                        buttons[-1].append(Int(n))

            assert items[-1].startswith('{')
            jolt_goal: List[Int] = []
            for n in items[-1][byte=1:-1].split(','):
                jolt_goal.append(Int(n))

            line: Line = Line(goal^, jolt_goal^, buttons^)
            min_clicks = line.min_clicks_jolt2()
            total += min_clicks
            line.print()
            print(min_clicks)
    print(total)

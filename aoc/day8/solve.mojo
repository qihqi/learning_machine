comptime fname = 'day8/input_large'
comptime iteration_num = 10
from std.collections import Set, Dict


def distance(first: Tuple[Int, Int, Int], second: Tuple[Int, Int, Int]) -> Int:
    x1, y1, z1 = first
    x2, y2, z2 = second

    total = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
    return total


struct UnionFind:
    var upper: List[Int]

    var to_group: Dict[Int, Set[Int]]

    def __init__(out self, size: Int):
        self.upper = List[Int](range(size))

        self.to_group = {}

    def find(self, x: Int) -> Int:
        head = x
        while head != self.upper[head]:
            head = self.upper[head]
        # print('find {} is {}'.format(x, head))
        return head

    def is_together(self, x: Int, y: Int)->Bool:
        return self.find(x) == self.find(y)

    def union(mut self, x: Int, y: Int) raises:
        headx = self.find(x)
        heady = self.find(y)

        self.upper[headx] = heady

        if heady not in self.to_group:
            self.to_group[heady] = {heady}

        if headx in self.to_group:
            groupx = self.to_group.pop(headx, {})
            self.to_group[heady].update(groupx)
        else:
            self.to_group[heady].add(headx)



def main() raises:
    coords: List[Tuple[Int, Int, Int]] = []
    with open(fname, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.strip():
                coo = line.split(',')
                coords.append((Int(coo[0]), Int(coo[1]), Int(coo[2])))

    # distance, first id, second id
    pairwise_dist = List[Tuple[Int, Int, Int]]()

    for i in range(len(coords)):
        for j in range(i):
            pairwise_dist.append(
                (distance(coords[i], coords[j]),
                i,
                j)
            )

    fn cmp_fn(a: Tuple[Int, Int, Int], b: Tuple[Int, Int, Int]) capturing -> Bool:
        return a[0] < b[0]
    sort[cmp_fn](Span(pairwise_dist))

    var union: UnionFind = UnionFind(len(coords))
    count = 0
    final_prod = 0
    edges = 0
    for _, x, y in pairwise_dist:

       # if count == iteration_num:
       #     break
        count += 1
        if union.is_together(x, y):
            print(x, y, 'is together')
            continue


        print('== union of {} and {} =='.format(x, y))
        union.union(x, y)
        edges += 1
        if len(coords) == 1 + edges:
            print('final prod elem', x, y, coords[x], coords[y])
            final_prod = coords[x][0] * coords[y][0]
            break

    sizes = List[Int]()
    for item in union.to_group.items():
        sizes.append(len(item.value))
    sort(sizes)
    print(sizes[-1] * sizes[-2] * sizes[-3])
    print(final_prod)

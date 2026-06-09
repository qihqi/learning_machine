
comptime fname = 'day4/input_small'
comptime MapT = List[List[UInt8]]

def _count_neighbor(themap: MapT, x: Int, y: Int) -> Int:
    length = len(themap)
    width = len(themap[0])
    total = 0
    for mv in  [(0, 1), (1, 0), (-1, 0), (0, -1),
                 (1, -1), (-1, -1), (-1, 1), (1, 1)]:
        newx = x + mv[0]
        newy = y + mv[1]
        if 0 <=newx < length and 0 <=newy < width:
            if themap[newx][newy] == UInt8(ord('@')):
                total += 1
    return Int(total < 4)


def count_neighbors_less_than_four(themap: MapT) -> List[Tuple[Int, Int]]:
    res: List[Tuple[Int, Int]] = []
    for x in range(len(themap)):
        for y in range(len(themap[0])):
            if themap[x][y] == UInt8(ord('@')):
                is_acc = _count_neighbor(themap, x, y)
                if is_acc == 1:
                    res.append((x, y))

    return res^

def remove(mut themap: MapT, idx: List[Tuple[Int, Int]]):
    for x, y in idx:
        themap[x][y] = UInt8(ord('.'))



def main() raises:
    themap: List[List[UInt8]] = []
    with open(fname, 'r') as f:
        raw_data = f.read()
        for line in raw_data.split('\n'):
            if line:
                themap.append(List[UInt8](line.as_bytes()))
    total = 0
    while True:
        to_remove = count_neighbors_less_than_four(themap)
        total += len(to_remove)
        if to_remove:
            remove(themap, to_remove)
        else:
            break
    print(total)

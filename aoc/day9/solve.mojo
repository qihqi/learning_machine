comptime fname = 'day9/input_large'
comptime GREEN:UInt8 = 1
comptime RED: UInt8 = 2

# 4616586187 is not right


def fill_row(mut row: List[UInt8]):

    first = len(row)
    last = 0
    seen = False
    for i in range(len(row)):
        if row[i] != 0 :
            last = i
        if not seen and row[i] != 0:
            first = i
            seen = True

    print('fill row', first, last)
    for i in range(first + 1, last):
        if row[i] == 0:
            row[i] = GREEN




def largest_rectangle(coords: List[Tuple[Int, Int]]) -> Int:
    so_far = 0
    for i in range(len(coords)):
        for j in range(i):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
            if area > so_far:
                so_far = area
                print('area {}, points {} {}, {} {}'.format(area, x1, y1, x2, y2))
    return so_far

def print_map(map:List[List[UInt8]]):
    for line in map:
        for elem in line:
            if elem == 0:
                print('.', end='')
            elif elem == GREEN:
                print('X', end='')
            elif elem == RED:
                print('#', end='')
        print()



def largest_green(coords: List[Tuple[Int, Int]]) -> Int:

    smallest_x = 2000000000
    smallest_y = 2000000000
    largest_x = 0
    largest_y = 0
    for y, x in coords:
        smallest_x = min(smallest_x, x)
        smallest_y = min(smallest_y, y)
        largest_x = max(largest_x, x)
        largest_y = max(largest_y, y)

    map_height = largest_x - smallest_x + 1
    map_width = largest_y - smallest_y + 1

    print( 'map size ', map_height, map_width)

    map = List[List[UInt8]]()
    for x in range(map_height):
        map.append([])
        for _ in range(map_width):
            map[x].append(0)

    for i, (y, x) in enumerate(coords):
        map[x - smallest_x][y - smallest_y] = RED

        nextx: Int
        nexty: Int
        if i == len(coords) - 1:
            nexty = coords[0][0]
            nextx = coords[0][1]
        else:
            nexty = coords[i + 1][0]
            nextx = coords[i + 1][1]

        if nextx == x:
            for j in range(min(y, nexty) + 1, max(y, nexty)):
                if map[x - smallest_x][ j - smallest_y] == 0:
                    map[x - smallest_x][ j - smallest_y] = GREEN
        elif nexty == y:
            for j in range(min(x, nextx) + 1, max(x, nextx)):
                if map[j - smallest_x][ y - smallest_y] == 0:
                    map[j - smallest_x][ y - smallest_y] = GREEN
        else:
            assert False


    # print_map(map)
    print('===============')

    for x in range(map_height):
        fill_row(map[x])

    #print_map(map)

    so_far = 0
    def is_inside(point: Tuple[Int, Int]) capturing -> Bool:
        x, y =  point[0] - smallest_x, point[1] - smallest_y
        return map[x][y] != 0

    def is_all_inside(x1: Int, y1: Int, x2: Int, y2: Int) capturing -> Bool:
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        for i in range(ymin, ymax + 1):
            if not is_inside((x1, i)):
                return False
            if not is_inside((x2, i)):
                return False

        for j in range(xmin, xmax+1):
            if not is_inside((j, y1)):
                return False
            if not is_inside((j, y2)):
                return False

        return True




    for i in range(len(coords)):
        for j in range(i):
            y1, x1 = coords[i]
            y2, x2 = coords[j]

            assert is_inside((x1, y1))
            assert is_inside((x2, y2))

            if is_all_inside(x1, y1, x2, y2):
                area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
                if area > so_far:
                    so_far = area
                    # print('area {}, points {} {}, {} {}'.format(area, x1, y1, x2, y2))

    return so_far


def main() raises:
    coords: List[Tuple[Int, Int]] = []
    with open(fname, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.strip():
                coo = line.split(',')
                coords.append((Int(coo[0]), Int(coo[1])))

    print(largest_green(coords))

comptime fname = 'day6/input_large'

def do_math2(all_lines: List[String]) raises -> Int:
    total = 0
    line_length = len(all_lines[0])
    current: Int = 0
    prev_op = ""
    for line in all_lines:
        line_length = max(line_length, len(line))
    for i in range(line_length):

        if i < len(all_lines[-1]):
            op = String(all_lines[-1][byte=i])
        else:
            op = ' '
        if op == '*':
            print('accumulating ', current)
            total += current
            current = 1
            prev_op = String(op)
        if op == '+':
            print('accumulating ', current)
            total += current
            current = 0
            prev_op = String(op)
        nums: String = ""

        for j in range(len(all_lines) - 1):
            if i < len(all_lines[j]) and String(all_lines[j][byte=i]) != ' ':
                nums += all_lines[j][byte=i]

        if nums:
            print(prev_op, nums)
            if prev_op == '*':
                current *= Int(nums)
            if prev_op == '+':
                current += Int(nums)

        if i == line_length - 1:
            print('accumulating ', current)
            total += current
    return total





def do_math(numbers: List[List[Int]], ops: List[String]) -> Int:
    print(len(ops), len(numbers))
    print(len(numbers[0]))
    total = 0
    for col in range(len(ops)):
        subtotal: Int  = 0
        if ops[col] == '*':
            subtotal = 1
        for row in range(len(numbers)):
            assert len(numbers[row]) > col, 'too short'
            if ops[col] == '*':
                subtotal *= numbers[row][col]
            else:
                subtotal += numbers[row][col]
        total += subtotal
    return total



def main() raises:
    numbers = List[List[Int]]()
    ops = List[String]()
    raw_lines = List[String]()
    with open(fname, 'r') as f:
        content = f.read()
        finish = False
        for line in content.split('\n'):
            if not line:
                continue
            raw_lines.append(String(line))
            if not finish:
                numbers.append([])
            for atom in line.split():
                try:
                    numbers[-1].append(Int(atom))
                except:
                    ops.append(String(atom))
                    finish = True
    _ = numbers.pop()
    #print(do_math(numbers, ops))
    print(do_math2(raw_lines))

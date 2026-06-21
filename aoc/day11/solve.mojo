comptime fname = 'day11/input_large'

def num_paths(mut mem: Dict[String, Int], out_dict: Dict[String, List[String]], key: String) raises -> Int:
    if key in mem:
        return mem[key]

    res = 0
    if key in out_dict:
        for n in out_dict[key]:
            res += num_paths(mem, out_dict, n)

    mem[key] = res
    return res

def main() raises:

    out_dict: Dict[String, List[String]] = {}

    with open(fname, 'r') as f:
        content = f.read()

        for line in content.split('\n'):
            if not line:
                continue
            items = line.split()

            out_dict[String(items[0][byte=:-1])] = []
            for i in items[1:]:
                out_dict[String(items[0][byte=:-1])].append(String(i))

    mem: Dict[String, Int] = {}
    mem["out"] = 1
    result = num_paths(mem, out_dict, "svr")
    dac_to_out = num_paths(mem, out_dict, "dac")
    fft_to_out = num_paths(mem, out_dict, "fft")
    print(result)

    print('===')
    mem = {}
    mem["fft"] = 1
    mem["out"] = 0
    srv_to_fft = num_paths(mem, out_dict, "svr")
    dac_to_fft = num_paths(mem, out_dict, "dac")

    mem = {}
    mem["dac"] = 1
    mem["out"] = 0
    srv_to_dac = num_paths(mem, out_dict, "svr")
    fft_to_dac = num_paths(mem, out_dict, "fft")

    print(srv_to_dac * dac_to_fft * fft_to_out + srv_to_fft * fft_to_dac * dac_to_out)

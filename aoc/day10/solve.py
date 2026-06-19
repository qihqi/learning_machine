import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
fname = 'day10/input_large'


def solve(buttons, goal):
    # button is List of List
    # goal is List
    #
    dim = len(goal)
    vecs = []
    for b in buttons:
        val = np.zeros((dim, ))
        val[b] = 1
        vecs.append(val)
    bmat = np.stack(vecs).T
    goal_vec = np.array(goal)
    c = np.ones((len(buttons), ))
    print(goal_vec.shape)
    print(c.shape)
    print(bmat.shape)
    ans = milp(
        c,
        integrality=1,
        bounds=Bounds(
            lb=0,
        ),
        constraints=LinearConstraint(
            bmat, lb=goal_vec, ub=goal_vec
        )
    )

    print(ans)
    return ans.fun



def main():
    total = 0
    with open(fname) as f:
        for line in f.readlines():
            if not line:
                continue
            tokens = list(line.split())
            buttons = []
            for b in tokens[1:-1]:
                numbers = list(map(int, b[1:-1].split(',')))
                print('button', numbers)
                buttons.append(numbers)
            goal = list(map(int, tokens[-1][1:-1].split(',')))
            res = solve(buttons, goal)
            total += res
    print(total)


if __name__ == '__main__':
    main()

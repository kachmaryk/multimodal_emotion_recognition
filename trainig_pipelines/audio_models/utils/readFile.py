from typing import List


def readFile(path) -> List:
    with open(path, 'r') as f:
        r = f.read()
        r = r.replace('=', '+').replace('\n', '+').split('+')

        new_r = []
        for i in r:
            if i == 'True':
                new_r.append('1')
            elif i == 'False':
                new_r.append(0)
            elif i != '' and '#' not in i:
                new_r.append(i)

    return new_r

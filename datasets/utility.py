def read_config(path: str) -> tuple:
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            if line[0] == '#':
                continue
            data.append(int(line))
    return data[0], data[1], data[2]
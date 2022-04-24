def w_sum(a, b):
    assert (len(a) == len(b))

    output = 0

    for j in range(len(a)):
        output += a[j] * b[j]

    return output


if __name__ == '__main__':
    a = [0, 1, 0, 1]
    b = [1, 0, 1, 0]
    c = [0, 1, 1, 0]
    d = [0.5, 0, 0.5, 0]
    e = [0, 1, -1, 0]

    print(w_sum(a, b))
    print(w_sum(b, c))
    print(w_sum(b, d))
    print(w_sum(c, c))
    print(w_sum(d, d))
    print(w_sum(c, e))


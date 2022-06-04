def w_sum(a, b):
    assert (len(a) == len(b))

    output = 0

    for j in range(len(a)):
        output += a[j] * b[j]

    # print("w_sum: a=" + str(a) + " b=" + str(b) + " = " + str(output))
    return output


def vect_mat_mul(v, m):
    assert (len(v) == len(m))

    output = [0, 0, 0]

    # print("v " + str(v))

    for j in range(len(v)):
        output[j] += w_sum(v, m[j])
        # print(str(j) + ": " + str(output))


    return output


def neural_network(input_data, weight_data):
    pred = vect_mat_mul(input_data, weight_data)
    return pred


if __name__ == '__main__':
    weights = [[0.1, 0.1, -0.3],
               [0.1, 0.2, 0],
               [0, 1.3, 0.1]]

    #        0      1    2    3
    games = [8.5 , 9.5, 9.9, 9.0]
    wins =  [0.65, 0.8, 0.8, 0.9]
    fans =  [1.2 , 1.3, 0.5, 1.0]

    for i in range(len(wins)):
        input_params = [
            games[i],
            wins[i],
            fans[i]
        ]
        prediction = neural_network(input_params, weights)
        print(str(i) + ":\n " + str(prediction))


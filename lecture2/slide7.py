def ele_mul(num, v):
    output = [0, 0, 0]

    assert (len(output) == len(v))

    for j in range(len(v)):
        output[j] += (num * v[j])
        #  print(output)

    return output


def neural_network(input_data, weight_data):
    pred = ele_mul(input_data, weight_data)
    return pred


if __name__ == '__main__':
    weights = [0.1, 0.5, 0.7]

    wins = [0.65, 0.8, 0.8, 0.9]

    for i in wins:
        prediction = neural_network(i, weights)
        print(prediction)


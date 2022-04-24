def w_sum(a, b):
    assert (len(a) == len(b))

    output = 0

    for j in range(len(a)):
        output += a[j] * b[j]

    return output


def neural_network(input_data, weight_data):
    pred = w_sum(input_data, weight_data)
    return pred


if __name__ == '__main__':
    weights = [
        0.05,  # koef for games
        0.15,  # koef for wins
        0.05   # koef for fans
    ]

    games = [8.5, 9.5, 9.9, 9.0]
    wins = [0.65, 0.8, 0.8, 0.9]
    fans = [1.2, 1.3, 0.5, 1.0]

    for i in range(len(games)):
        input_params = [
            games[i],
            wins[i],
            fans[i]
        ]
        prediction = neural_network(input_params, weights)
        print(prediction)


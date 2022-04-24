import numpy as np


def neural_network(input_data, weight_data):
    pred = input_data.dot(weight_data)
    return pred


if __name__ == '__main__':
    weights = np.array([
        0.05,  # koef for games
        0.15,  # koef for wins
        0.05   # koef for fans
    ])

    games = np.array([8.5, 9.5, 9.9, 9.0])
    wins = np.array([0.65, 0.8, 0.8, 0.9])
    fans = np.array([1.2, 1.3, 0.5, 1.0])

    for i in range(len(games)):
        input_params = np.array([
            games[i],
            wins[i],
            fans[i]
        ])
        prediction = neural_network(input_params, weights)
        print(prediction)


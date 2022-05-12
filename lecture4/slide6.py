import matplotlib.pyplot as plt
import numpy as np


def neural_network(input_param, weight_param):
    return input_param.dot(weight_param)


if __name__ == '__main__':

    goal_prediction = 1

    input_params = np.array([
        8.5,   # amount of games
        -0.65  # perc of wins
    ])
    print("input_params", input_params)

    weights = np.array([
        0.01,   # amount of games
        0.02   # perc of wins
    ])
    print("initial_weights: ", weights)

    alpha = 0.005

    x_weight_games = []
    x_weight_wins = []
    y_error = []

    for i in range(1000):
        print("------ loop ", i + 1)

        print("weights: ", weights)
        x_weight_games.append(weights[0])
        x_weight_wins.append(weights[1])

        prediction = neural_network(input_params, weights)

        print("prediction: ", prediction)

        error = (prediction - goal_prediction) ** 2
        y_error.append(error)

        print("error: ", error)

        delta = prediction - goal_prediction

        print("delta: ", delta)

        weight_deltas = input_params * delta
        # hold games weight
        weight_deltas[0] = 0

        print("weight_deltas: ", weight_deltas)

        weights = weights - (weight_deltas * alpha)

        print("new_weights: ", weights)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_weight_games, x_weight_wins, y_error, '-')
    ax.set_title('Gradient descent')
    ax.set_xlabel("games")
    ax.set_ylabel("win")
    ax.set_zlabel("error")
    plt.show()


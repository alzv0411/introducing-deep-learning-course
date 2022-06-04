import matplotlib.pyplot as plt
import numpy as np


def neural_network(input_param, weight_param):
    return input_param.dot(weight_param)


if __name__ == '__main__':

    goal_prediction = 1

    input_params = np.array([
        8.5,   # amount of games
        0.65,  # perc of wins,
        1.2,   # amount of fans
    ])
    print("input_params", input_params)

    weights = np.array([
        0.01,   # amount of games
        0.02,   # perc of wins,
        0.015,  # amount of fans
    ])
    print("initial_weights: ", weights)

    alpha = 0.01

    x_weight_games = []
    x_weight_wins = []
    x_weight_fans = []
    y_error = []

    for i in range(1000):
        print("------ loop ", i + 1)

        print("weights: ", weights)
        x_weight_games.append(weights[0])
        x_weight_wins.append(weights[1])
        x_weight_fans.append(weights[2])

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

    plt.xlabel('weight')
    plt.ylabel('error')

    plt.axhline(y=0, color="grey", linestyle="--")

    plt.plot(x_weight_games, y_error, 'b-', label="weight_games=" + str(weights[0]))
    plt.plot(x_weight_wins, y_error, 'r-', label="weight_wins=" + str(weights[1]))
    plt.plot(x_weight_fans, y_error, 'g-', label="weight_fans" + str(weights[2]))

    plt.title('Gradient descent')
    plt.legend()

    plt.show()


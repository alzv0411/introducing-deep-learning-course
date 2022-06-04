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
        0.1,   # amount of games
        0.2,   # perc of wins,
        -0.1,  # amount of fans
    ])

    alpha = 0.01

    for i in range(1000):
        print("------ loop ", i + 1)

        print("weights: ", weights)

        prediction = neural_network(input_params, weights)

        print("prediction: ", prediction)

        error = (prediction - goal_prediction) ** 2

        print("error: ", error)

        delta = prediction - goal_prediction

        print("delta: ", delta)

        weight_deltas = input_params * delta
        # hold games weight
        weight_deltas[0] = 0

        print("weight_deltas: ", weight_deltas)

        weights = weights - (weight_deltas * alpha)

        print("new_weights: ", weights)

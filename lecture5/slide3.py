import numpy as np


def neural_network(input_param, weight_param):
    return input_param * weight_param


if __name__ == '__main__':

    input_win = 0.65
    print("input_win", input_win)

    weights = np.array([
        0.3,   # hurt
        0.2,   # wins
        0.9,   # sad
    ])

    alpha = 0.1

    goal_prediction = np.array([
        0.1,   # hurt
        1,   # wins
        0.1,   # sad
    ])
    print("goal_prediction", goal_prediction)

    for i in range(50):
        print("------ loop ", i + 1)
        print("weights: ", weights)

        prediction = neural_network(input_win, weights)
        print("prediction: ", prediction)
        error = (prediction - goal_prediction) ** 2
        print("error: ", error)
        error_sum = error.sum()
        print("error.sum: ", error_sum)
        delta = prediction - goal_prediction
        print("delta: ", delta)
        weight_deltas = input_win * delta
        print("weight_deltas: ", weight_deltas)
        weight_deltas_alpha = weight_deltas * alpha
        print("weight_deltas_alpha: ", weight_deltas_alpha)
        weights = weights - weight_deltas_alpha
        print("new_weights: ", weights)

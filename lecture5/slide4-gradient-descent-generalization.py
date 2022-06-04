import numpy as np


def neural_network(input_param, weights_param):
    return weights_param.dot(input_param)


if __name__ == '__main__':

    np.set_printoptions(formatter={'float': '{: 1.4f}'.format})

    goal_prediction = np.array([
        0.1,  # hurt
        1,    # wins
        0.1,  # sad
    ])

    input_params = np.array([
        8.5,  # amount of games
        0.65,  # perc of wins,
        1.2,  # amount of fans
    ])
    print("input_params", input_params)

    initial_weights = np.array([
        # games # wins # fans
        [0.1,   0.1,   -0.3],  # hurt
        [0.1,   0.2,   0],     # win?
        [0.0,   1.3,   0.1]    # sad
    ])

    weights = np.copy(initial_weights)

    alpha = 0.01

    print("goal_prediction", goal_prediction)

    for i in range(16):
        print("------ loop ", i + 1)
        print("weights:\n", weights)

        prediction = neural_network(input_params, weights)  # Home work
        print("prediction: ", prediction)

        error = (prediction - goal_prediction) ** 2
        print("error: ", error)

        delta = prediction - goal_prediction
        print("delta: ", delta)

        weight_deltas = np.outer(delta, input_params)  # Home work
        print("weight_deltas:\n", weight_deltas)

        weights = weights - (weight_deltas * alpha)
        print("new_weights:\n", weights)

    print("\n\n\ninitial weights\n", initial_weights)
    print("corrected weights\n", weights)

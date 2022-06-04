import numpy as np


if __name__ == '__main__':
    # MNIST

    # HOME WORK:
    # 10 x 10 - A, B, C, D, E

    test_inputs = np.array([
        [
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,     # 100% => 0, 0% => 1, 0% => 2
            0, 1, 0, 1, 0,     # [1, 0, 0]
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
        ],
        [
            0, 0, 1, 0, 0,
            0, 1, 1, 0, 0,
            0, 0, 1, 0, 0,     # [0, 1, 0]
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            0, 1, 1, 1, 0,
            0, 0, 0, 1, 0,
            0, 0, 1, 0, 0,     # [0, 0, 1]
            0, 1, 0, 0, 0,
            0, 1, 1, 1, 0,
        ],
        [
            0, 1, 1, 1, 0,
            0, 0, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 1, 0,
            0, 1, 1, 1, 0,
        ]
    ])

    test_predictions = np.array([
        [1, 0, 0, 0],  # 0
        [0, 1, 0, 0],  # 1
        [0, 0, 1, 0],  # 2
        [0, 0, 0, 1]   # 3
    ])

    np.set_printoptions(formatter={'float': '{: 1.2f}'.format}, linewidth=13)
    print("test_inputs:\n", test_inputs, "\ntest_predictions:\n", test_predictions)

    initial_weights = np.full((test_predictions.shape[1], test_inputs.shape[1]), 0.01)

    np.set_printoptions(formatter={'float': '{: 1.2f}'.format}, linewidth=35)
    print("initial_weights: \n", initial_weights)

    alpha = 0.001

    weights = np.copy(initial_weights)

    min_error = 1
    min_i = 0

    print("Leaning...")
    for i in range(14325):
        all_error = 0
        for test_input, test_prediction in zip(test_inputs, test_predictions):
            prediction = weights.dot(test_input)
            # print("prediction: ", prediction)

            delta = prediction - test_prediction
            # print("delta: ", delta)

            weight_deltas = np.outer(delta, test_input)
            # print("weight_deltas:\n", weight_deltas)

            weights = weights - (weight_deltas * alpha)
            # print("new_weights:\n", weights)

            error = (prediction - test_prediction) ** 2
            # print("error: ", error)
            all_error += error.sum()

        if all_error < min_error:
            min_error = all_error
            min_i = i

    print("min error: ", min_error, " was at i=", min_i)

    np.set_printoptions(formatter={'float': '{: 1.2f}'.format}, linewidth=35)
    print("Corrected weights: \n", weights)

    inputs = np.array([
        [
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
        ],
        [
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            0, 1, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 1, 0,
        ],
        [
            0, 1, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 1, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 1, 1, 0, 0,
        ],
        [
            0, 0, 0, 1, 0,
            0, 0, 1, 1, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 1, 0,
        ]
    ])

    np.set_printoptions(formatter={'float': '{: 1.2f}'.format}, linewidth=11)
    for test_input in inputs:
        print("\nInput:\n", test_input)
        prediction = weights.dot(test_input)
        for i in range(len(prediction)):
            print("prediction for ", i, " is ", round(prediction[i], 2))

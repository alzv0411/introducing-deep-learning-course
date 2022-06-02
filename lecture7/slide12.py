import numpy as np


def relu(x):
    return (x > 0) * x


def relu2(output):
    return output > 0


if __name__ == '__main__':

    np.random.seed(1)

    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    streetlights = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    walk_stop_results = np.array([[
        1,
        1,
        0,
        0
    ]]).T

    alpha = 0.2
    hidden_size = 4

    print("streetlights:\n", streetlights, "\nwalk_stops:\n", walk_stop_results)

    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    print("weights_0_1:\n", weights_0_1, "\nweights_1_2:\n", weights_1_2)

    for it in range(10):
        layer_2_error = 0
        for i in range(len(streetlights)):
            layer_0 = streetlights[i:i+1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
            print("layer_0:\n", layer_0, "\nlayer_1:\n", layer_1, "\nlayer_2:\n", layer_2)

            layer_2_error += np.sum((layer_2 - walk_stop_results[i:i+1]) ** 2)

            layer_2_delta = walk_stop_results[i:i+1] - layer_2
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2(layer_1)  # !!!
            print("layer_2_delta:\n", layer_2_delta, "\nlayer_1_delta:\n", layer_1_delta)

            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            print("weights_0_1:\n", weights_0_1, "\nweights_1_2:\n", weights_1_2)

        print("Iteration: ", it, " error: ", layer_2_error)

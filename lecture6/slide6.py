import numpy as np

if __name__ == '__main__':
    streetlights = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    walk_stop_results = np.array([
        1,
        1,
        0,
        0
    ])

    weights = np.array([0.5, 0.5, 0.5])

    alpha = 0.1

    for i in range(600):
        sum_error = 0

        for streetlight, walk_stop in zip(streetlights, walk_stop_results):
            prediction = streetlight.dot(weights)

            error = (walk_stop - prediction) ** 2
            sum_error += error

            delta = prediction - walk_stop
            weights = weights - (alpha * (streetlight * delta))
            # print("prediction: ", prediction, " error: ", error, " delta: ", delta, " weights: ", weights)

        print("Iteration: ", i, " sum_error: ", sum_error, " weights: ", weights)

    for streetlight, walk_stop in zip(streetlights, walk_stop_results):
        prediction = streetlight.dot(weights)
        print("streetlight: ", streetlight, " prediction: ", prediction, " walk_stop: ", walk_stop)
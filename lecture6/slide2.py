import numpy as np

if __name__ == '__main__':

    np.set_printoptions(formatter={'float': '{: 1.4f}'.format})

    games = [8.5, 9.5, 9.9, 9.0]
    total_wins = [0.65, 0.8, 0.8, 0.9]
    fans = [1.2, 1.3, 0.5, 1.0]

    hurt_test = [0.1, 0.0, 0.0, 0.1]
    win_test = [1, 1, 0, 1]
    sad_test = [0.1, 0.0, 0.1, 0.2]

    initial_weights = np.array([
        # games # wins # fans
        [0.1,   0.1,   -0.3],  # hurt
        [0.1,   0.2,   0],     # win?
        [0.0,   1.3,   0.1]    # sad
    ])

    weights = np.copy(initial_weights)
    best_weights = np.copy(initial_weights)

    alpha = 0.001

    min_error = 1
    min_i = 0

    print("Leaning...")
    for i in range(208849):
        all_error = 0
        for game, total_win, fan, hurt, win, sad in zip(games, total_wins, fans, hurt_test, win_test, sad_test):
            input_params = np.array([game, total_win, fan])

            prediction = weights.dot(input_params)
            # print("prediction: ", prediction)

            goal_prediction = np.array([hurt, win, sad])
            # print("goal_prediction: ", goal_prediction)

            delta = prediction - goal_prediction
            # print("delta: ", delta)

            weight_deltas = np.outer(delta, input_params)  # Home work
            # print("weight_deltas:\n", weight_deltas)

            weights = weights - (weight_deltas * alpha)
            # print("new_weights:\n", weights)

            error = (prediction - goal_prediction) ** 2
            # print("error: ", error)
            all_error += error.sum()

        # print("error: ", all_error)
        if all_error < min_error:
            min_error = all_error
            min_i = i
            best_weights = np.copy(weights)

    print("min error: ", min_error, " was at i=", min_i)

    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print("\nInitial\n", initial_weights, " \nBest\n", best_weights)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("Test: input -> prediction -> goal_pred -> error")
    for game, total_win, fan, hurt, win, sad in zip(games, total_wins, fans, hurt_test, win_test, sad_test):
        goal_prediction = np.array([hurt, win, sad])
        input_params = np.array([game, total_win, fan])
        prediction = best_weights.dot(input_params)
        error = (prediction - goal_prediction) ** 2

        print(input_params, " -> ", prediction, " -> ", goal_prediction, " -> ", error)

    # input_params = np.array([
    #     10,
    #     0.1,
    #     0.3
    # ])
    # prediction = best_weights.dot(input_params)
    # print(input_params, " -> ", prediction)



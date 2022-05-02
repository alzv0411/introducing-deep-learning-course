def neural_network(input_data, weight_data):
    return input_data * weight_data


if __name__ == '__main__':

    weight = 0.5

    input_value = 0.5

    goal_prediction = 0.8

    lr = 0.2

    for i in range(40):
        prediction = neural_network(input_value, weight)
        error = (prediction - goal_prediction) ** 2

        if error < 0.05:
            lr = 0.05

        if error < 0.01:
            lr = 0.01

        print("error: ", error,
              "prediction: ", round(prediction, 4),
              " weight: ", round(weight, 4))

        p_up = neural_network(input_value, weight + lr)
        e_up = (p_up - goal_prediction) ** 2

        p_dn = neural_network(input_value, weight - lr)
        e_dn = (p_dn - goal_prediction) ** 2

        if e_dn < e_up:
            weight -= lr

        if e_up < e_dn:
            weight += lr


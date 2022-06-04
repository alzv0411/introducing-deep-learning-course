def neural_network(input_data, weight_data):
    return input_data * weight_data


if __name__ == '__main__':

    weight = 0.0

    input_value = 1.1

    goal_prediction = 0.8

    for i in range(4):
        prediction = neural_network(input_value, weight)
        error = (prediction - goal_prediction) ** 2

        delta = prediction - goal_prediction

        weigh_delta = input_value * delta

        weight = weight - weigh_delta
        print("--------")
        print("error: ", error,
              "\nprediction: ", round(prediction, 4),
              "\ndelta: ", round(delta, 4),
              "\nweigh_delta: ", round(weigh_delta, 4),
              "\nweight: ", round(weight, 4))



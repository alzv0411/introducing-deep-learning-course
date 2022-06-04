def neural_network(input_data, weight_data):
    return input_data * weight_data


if __name__ == '__main__':

    weight = 0.5

    input_value = 2

    goal_prediction = 0.8

    alpha = 0.1

    for i in range(20):
        print("-------- current weight: ", round(weight, 4))

        prediction = neural_network(input_value, weight)
        error = (prediction - goal_prediction) ** 2

        delta = prediction - goal_prediction

        weigh_delta = input_value * delta

        weight = weight - (weigh_delta * alpha)

        print("error: ", error,
              "\nprediction: ", round(prediction, 4),
              "\ndelta: ", round(delta, 4),
              "\nweigh_delta: ", round(weigh_delta, 4),
              "\nnew weight: ", round(weight, 4))


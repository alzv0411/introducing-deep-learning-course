if __name__ == '__main__':
    weight = 0.5
    input_value = 0.5

    goal_pred = 0.8

    pred = input_value * weight

    print("pred=", pred)

    print(pred - goal_pred)

    error = (pred - goal_pred) ** 2

    print(round(error, 4))


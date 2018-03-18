from nn import NeuralNetwork
from pre_process import process_iris_data
from gui import train_with_sim


if __name__ == "__main__":
    hidden_layer_list = [6, 8, 6]
    X_train, y_train, X_test, y_test = process_iris_data()
    NN = NeuralNetwork(4, hidden_layer_list, 3)
    train_with_sim(NN, X_train, y_train, 10000000000)

import numpy as np
import matplotlib.pyplot as plt

"""fixme --> Primeiro passo para melhorar o aprendizado da nossa rede neural:
alteramos nossos pesos e bias para numeros aleatórios. 
porém proximos a zero, por isso multiplicamos por 0.01        
self.weights = np.zeros(num_imputs)
self.bias = 0.0
(13-06-23)

validação e pré-processamento de dados
antes de chamar o método train
verificamos se os conjuntos de treinamento e rótulos tem o mesmo
numero de amostras e estão em um formato apropriado
(asset len)
(16/06/23)

Avaliação de desempenho adicionado como método
avalia o desempenho da rede neural perceptron
(16/06/23)

Criação de plots para vizualização de dados
utilizando a biblioteca matplotlib
(18/06/2023)


adicionar o grid-search para hiperparâmetros ??

"""


class Perceptron:
    def __init__(self, num_inputs, initial_learning_rate=0.01, decay_rate=0.1, activation_fn=None):
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = np.random.rand() * 0.01
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.activation_fn = activation_fn if activation_fn is not None else np.heaviside

    def predict(self, inputs):
        activation = np.dot(inputs, self.weights) + self.bias
        return self.activation_fn(activation, 0.5)

    def train(self, training_inputs, labels, num_epochs):
        assert len(training_inputs) == len(labels), "Número de entradas de treinamento e rótulos não correspondem"

        learning_rate = self.initial_learning_rate
        for epoch in range(1, num_epochs + 1):
            learning_rate *= (1.0 / (1.0 + self.decay_rate * epoch))
            print("Epoch:", epoch)
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                print("Inputs:", inputs, "Predicted:", prediction)
                self.weights += learning_rate * (label - prediction) * inputs
                self.bias += learning_rate * (label - prediction)

    def evaluate(self, test_inputs, test_labels):
        correct_predictions = 0
        total_predictions = len(test_inputs)
        for inputs, label in zip(test_inputs, test_labels):
            prediction = self.predict(inputs)
            if prediction == label:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        return accuracy


def main():
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 0]])
    # Adicione um rótulo para cada entrada de treinamento
    labels = np.array([0, 0, 0, 1, 0, 1])

    perceptron = Perceptron(num_inputs=2, initial_learning_rate=0.1)
    perceptron.train(training_inputs, labels, num_epochs=10)

    test_inputs = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 0]])
    for inputs in test_inputs:
        prediction = perceptron.predict(inputs)
        print(f"Inputs: {inputs} Predicted: {prediction}")

    plt.figure(figsize=(8, 6))

    # rotulo para as cores
    color_map = {0: 'blue', 1: 'red'}

    colors = [color_map[int(prediction)] for _ in range(len(test_inputs))]

    plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=colors)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Predictions')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()

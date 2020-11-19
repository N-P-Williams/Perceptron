import numpy as np


class Perceptron(object):

    def __init__(self, num_of_inputs, threshold=100, learn_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learn_rate
        self.weights = np.zeros(num_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels_AND = np.array([1, 0, 0, 0])
labels_OR = np.array([1, 1, 1, 0])

if __name__ == "__main__":
    perceptron_AND = Perceptron(2)
    perceptron_OR = Perceptron(2)
    perceptron_AND.train(training_inputs, labels_AND)
    perceptron_OR.train(training_inputs, labels_OR)
    inputs = np.array([1, 1])
    # take the and and NOT it
    xor_and = perceptron_AND.predict(inputs)
    if(xor_and):
        xor_and = 0
    else:
        xor_and = 1
    # take the OR
    xor_or = perceptron_OR.predict(inputs)
    # make them into a input
    inputs_XOR = np.array([xor_and, xor_or])
    print("inputs:", inputs)
    print("AND:", perceptron_AND.predict(inputs))
    print("OR: ", perceptron_OR.predict(inputs))
    print("XOR:", perceptron_AND.predict(inputs_XOR))

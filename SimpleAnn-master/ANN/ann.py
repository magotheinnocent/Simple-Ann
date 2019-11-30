import numpy as np
import copy


class ANN:
    def __init__(self, learning_rate, weights):
        self.weights = weights
        self.learning_rate = learning_rate

        self.outputs = []

    def train(self, input_data, output_target_data):
        for i, (row, t) in enumerate(zip(input_data, output_target_data)):
            print('________EPOCH: {}________'.format(i+1))
            output = self.__feedfoward(row)
            error = self.__outputerror(*output, t)
            self.__adjust_weights(error)
            print('\n')

    def __feedfoward(self, input_data_row):
        outputs = input_data_row
        for w in self.weights:
            outputs = np.dot(w, outputs)
            self.outputs.append(outputs)
            outputs = [self.__sigmoid(x) for x in outputs]
        return outputs

    def __outputerror(self, modeloutput, actualoutput):
        return (actualoutput - modeloutput) * modeloutput * (1 - modeloutput)

    def __adjust_weights(self, output_error):
        weights_copy = copy.deepcopy(self.weights)
        for idx_1, layer in enumerate(reversed(self.weights)):
            for idx_2, n in enumerate(layer):
                output = self.outputs[-(idx_1 + 1)][idx_2]
                if idx_1 > 0:
                    weight = self.weights[-1][0][idx_2]
                    output_error = output_error * weight * output * (1 - output)

                for idx_3, w in enumerate(n):
                    print('for weight', weights_copy[-(idx_1+1)][idx_2][idx_3])
                    weights_copy[-(idx_1+1)][idx_2][idx_3] += self.learning_rate * output_error * output
                    print(weights_copy)

        self.weights = weights_copy


    @staticmethod
    def __sigmoid(num):
        return 1/(1 + np.exp(-num))
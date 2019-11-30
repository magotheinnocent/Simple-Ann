from unittest import TestCase
import pandas as pd
from ANN.ann import ANN
from util.math_ops import bind_between_0_1


class TestANN(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        data = pd.read_csv(
            "/home/john-gachihi/PycharmProjects/SimpleANN/training_data.csv", header=None)

        data = data.applymap(lambda value: bind_between_0_1(
            value, data.max().max(), data.min().min()))

        cls.output_target = data.iloc[:, -1].values
        cls.input_data = data.drop(3, axis=1).values
        cls.ann = ANN(0.5, weights=[[[0.2, 0.3, 0.2], [0.1, 0.1, 0.1]], [[0.5, 0.1]]])

    def test_train(self):
        self.ann.train(self.input_data, self.output_target)
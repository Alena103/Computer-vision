# Импортируем neuron, math в train
from pure_puthon.neuron import Neuron
from math import pow

class OneLayerNet:
    # Количество входных и выходых нейронов
    def __init__(self, inputs_count, output_neurons_count):
        self.__inputs_count = inputs_count
        self.__neurons = []
        for j in range(output_neurons_count):
            self.__neurons.append(Neuron(inputs_count))


    def train(self, vector, learning_rate):
        # Начинается перебор обучающих векторов
        for j in range(len(self.__neurons)):
            # Из вектора берем обучающий  вектор, calc_y-вычисленный вес в нейроне
            self.__neurons[j].calc_y(vector.get_x())
        # Вычисляем веса для дельт
        weights_deltas = [[0] * (len(vector.get_x()) + 1)] * len(self.__neurons)
        loss = 0
        # Перебираем все нейроны
        for j in range(len(self.__neurons)):
            sigma = (vector.get_d()[j] - self.__neurons[j].get_y()) \
                    * self.__neurons[j].derivative()# Считаем для каждого нейрона сигму
            # Нулевой строке массива дельт весов присваиваем значение скорости обучения на сигму нейрона
            weights_deltas[j][0] = learning_rate * sigma
            # Количество весов у контретного нейрона
            wlen = len(self.__neurons[j].get_weights())
            # Далее проходим по всем весам каждой строки
            for i in range(wlen):
                # Считаем новые веса для каждого элемента массива, например для первой
                # Строки умножаем на первое значение обучающего вектора, для второй строки и тд
                weights_deltas[j][i] = learning_rate * sigma * vector.get_x()[i]
            # Для каждого нейрона корректируем вес тем,что прибавляем получившуюся дельту вусов
            self.__neurons[j].correct_weights(weights_deltas[j])
            # Вычисляем сумму  ошибок для каждого нейрона
            loss += pow(vector.get_d()[j] - self.__neurons[j].get_y(), 2) #квадрат разности между желаймым и действительлным

        # Возвращаем половину ошибк
        return 0.5 * loss

    def test(self, vector):
        y = [0] * len(self.__neurons)
        for j in range(len(self.__neurons)):
            # Из вектора берем обучающий  вектор, calc_y
            self.__neurons[j].calc_y(vector.get_x())
            y[j] = self.__neurons[j].get_y()
        return y
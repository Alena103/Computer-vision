from tensor_flow import neural_util as nu


class OneLayerNet(object):
    # Функция, преобразующая входные аргументы x, num_classes в параметры класса
    def __init__(self, x, num_classes):
        self.X = x
        self.NUM_CLASSES = num_classes
        # Задаем парметр/функцию для построения графа
        self.output = self.create()

    # Функция постройки графа
    def create(self):
        return nu.fc(self.X, self.X.get_shape()[1], self.NUM_CLASSES, name='one_layer_perceptron')
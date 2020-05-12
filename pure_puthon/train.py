# Импортируем one_layer_net, datareader, vector, datetime в train
from pure_puthon.one_layer_net import OneLayerNet
from pure_puthon.datareader import DataReader
from pure_puthon._vector import Vector
from datetime import datetime
# numpy модуль , который предоставляет общие математические и числовые операции в виде пре-скомпилированных, быстрых функций
import numpy as np
# OpenCV библиотека алгоритмов компьютерного зрения, обработки изображений и численных алгоритмов общего назначения с открытым кодом.
import cv2

#получить максимальный нейрон
def get_max_neuron_idx(neurons):
    max_idx = -1
    answer = -1
    # Перебор всех нейронов
    for j in range(len(neurons)):
        if neurons[j] > answer:  
            answer = neurons[j]
            max_idx = j
    return max_idx

# Скоростные параметры
# Скорость обучения и количество эпох
learning_rate = 1e-6
num_epochs = 10
# Входной канал
input_channels = 1
# Высота
input_height = 28
# Ширина
input_width = 28
# Количество классов изображений
num_classes = 6
# Размер изображения, класс изображения(двойка , тройка и тд)
one_layer_net = OneLayerNet(input_height * input_width, num_classes)

# Путь к картинкам
train_dir = "data/train"
test_dir = "data/test"
# Берем с помощью DataReader изображения тренировочную и тестовую выборку
train_generator = DataReader(train_dir, [input_height, input_width], True, input_channels, num_classes).get_generator()
test_generator = DataReader(test_dir, [input_height, input_width], False, input_channels, num_classes).get_generator()

print('Вес тренировочных изображений: {}'.format(train_generator.get_data_size()))
print('Вес тесовых изображенй изображений: {}'.format(test_generator.get_data_size()))
# Текущее время начала обучения
print("{} Приступить к тренировкам...".format(datetime.now()))

# Основная функция, которая для каждой итерации считает ошибочность распознавания в обучении
for epoch in range(num_epochs):
    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
    loss = 0
    # Перебор векторов
    for m in range(train_generator.get_data_size()):
        x, d = train_generator.next()
        # Подсчет ошибок
        loss += one_layer_net.train(Vector(x, d), learning_rate)
    print("loss = {}".format(loss / train_generator.get_data_size()))
    train_generator.reset_pointer()
    train_generator.shuffle_data()
passed = 0

# Насколько ожидаемый результат совпадает с реально распознанным числом
for i in range(test_generator.get_data_size()): 
    x, d = test_generator.next()
    y = one_layer_net.test(Vector(x, d))
    d_max_idx = get_max_neuron_idx(d)
    y_max_idx = get_max_neuron_idx(y)
    if y_max_idx == d_max_idx:
        passed += 1
    print("{} признана как {}".format(d_max_idx, y_max_idx))

# Процент совпадения, где проверяются ожидаемый вектор и полученный вектор
accuracy = passed / test_generator.get_data_size() * 100.0
print("Точность: {:.4f}%".format(accuracy))
print("Распознавание пользовательского изображения")
# custom.bmp -входное изображение
img = cv2.imread("custom.bmp", cv2.IMREAD_GRAYSCALE)
# cv2.imread-импортирует изображение и просматривает, в кавычках указывается путь
img = img.reshape((img.shape[0], img.shape[1], 1)).astype(np.float32)
y = one_layer_net.test(Vector(img, None))
print("Пользовательское изображение распознается как {}".format(get_max_neuron_idx(y)))
# Изображение может распознааться как, наприме 3.
# format(get_max_neuron_idx(y))-цифра которой определилось озображение
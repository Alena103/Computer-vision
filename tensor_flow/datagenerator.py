import numpy as np
import cv2


class DataGenerator:
    def __init__(self, patterns, labels, scale_size, shuffle=False, input_channels=3, nb_classes=8):

        # Инициализация параметров
        self.__n_classes = nb_classes
        self.__shuffle = shuffle
        # Входные сигналы
        self.__input_channels = input_channels
        # Размер шкалы
        self.__scale_size = scale_size
        # Указатель
        self.__pointer = 0
        # Размер данных
        self.__data_size = len(labels)
        # Шаблоны
        self.__patterns = patterns
        self.__labels = labels
        
        if self.__shuffle:
            self.shuffle_data()

    def get_data_size(self):
        return self.__data_size
    # Случайное перемешивание изображений и меток
    def shuffle_data(self):
        images = self.__patterns.copy()
        labels = self.__labels.copy()
        self.__patterns = []
        self.__labels = []
        # Создадим список перестановочного индекса и перемешаем данные в соответствии с списком
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.__patterns.append(images[i])
            self.__labels.append(labels[i])
    # Сбросим указатель на начало списка
    def reset_pointer(self):
        self.__pointer = 0
        if self.__shuffle:
            self.shuffle_data()

    # Эта функция получает следующие n (= batch_size) изображений из
    # списка путей,маркирует и загружает изображения в них в память
    def next(self):
        # Получаем следующую партию изображения (путь) и метки
        path = self.__patterns[self.__pointer]
        label = self.__labels[self.__pointer]
        # Обновляем указатель
        self.__pointer += 1
        # Считываем шаблон
        if self.__input_channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)

        #  Масштабируем изображение.
        # Масштаби́рование изображения — изменение размера изображения с сохранением пропорций.
        img = cv2.resize(img, (self.__scale_size[0], self.__scale_size[1]))
        img = img.astype(np.float32)
        if self.__input_channels == 1:
            img = img.reshape((img.shape[0], img.shape[1], 1)).astype(np.float32)
        # Разверачиваем метки до одной хот кодировки
        # Возвращаем новый массив заданной формы и типа, заполненный нулями.nt  6
        one_hot_labels = np.zeros(self.__n_classes)
        one_hot_labels[label] = 1
        # Возвращаем массив изображений и меток
        return img, one_hot_labels
import numpy as np

#возвращает параметры х и desireOutputs в виде массива, внутри __init__делая его массивом нужного вида
class Vector:

    def __init__(self, x, desireOutputs):
        if len(x.shape) > 2:#x.shape-размер массива
            self.__x = np.asarray(x).reshape(-1)
            self.__x = self.__x.reshape(1, self.__x.shape[0]).astype(np.float32)#asanyarray преобразует данные в массив
            # reshape- изменяет форму массива без изменения данных
        self.__desireOutputs = desireOutputs.reshape(1, desireOutputs.shape[0])

    def get_x(self):
        return self.__x

    def get_desire_outputs(self):
        return self.__desireOutputs

import numpy as np
# В общем классе возвращаются параметры х и d в виде массива, внутри __init__делая его массивом нужного вида
class Vector:
    def __init__(self, x, d):
        # x.shape-размер массива
        if len(x.shape) > 2:
            # asanyarray преобразует данные в массив
            self.__x = list(np.asanyarray(x).reshape(-2))
            # reshape- изменяет форму массива без изменения данных
        else:
            self.__x = x
        self.__d = d

    def get_x(self):
        return self.__x

    def get_d(self):
        return self.__d
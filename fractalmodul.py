# ----- Построение фракталов с помощью систем сжимающих отображений --------
import numpy as np

"""
Для построения фракталов в качестве множеств сгущения будем использовать следующие множества точек,
описываемых в виде последовательности точек, соединяемых отрезками:
"""

treug = np.array([[0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0]])
kust =  np.array([[0, 0, 1.25*np.sqrt(2)/2, 0, -1.5*np.sqrt(2)/2],
                  [0, 0.5, 0.5+1.25*np.sqrt(2)/2, 0.5, 0.6+1.5*np.sqrt(2)/2]])
line0 = np.array([[0, 0], [0, 1]])
poligon6 = np.array([[-0.5, 0.5, 1, 0.5, -0.5, -1, -0.5],
                     [0, 0, np.sqrt(3)/2, np.sqrt(3), np.sqrt(3), np.sqrt(3)/2, 0]])
anglek = np.array([[0, 1, 1], [0, 0, 1]])
circle = np.array([[1, np.cos(np.pi/6), np.cos(np.pi*2/6), np.cos(np.pi*3/6), np.cos(np.pi*4/6),
                    np.cos(np.pi*5/6), np.cos(np.pi*6/6), np.cos(np.pi*7/6), np.cos(np.pi*8/6),
                    np.cos(np.pi*9/6), np.cos(np.pi*10/6), np.cos(np.pi*11/6), 1],
                   [0, np.sin(np.pi / 6), np.sin(np.pi * 2 / 6), np.sin(np.pi * 3 / 6), np.sin(np.pi * 4 / 6),
                    np.sin(np.pi * 5 / 6), np.sin(np.pi * 6 / 6), np.sin(np.pi * 7 / 6), np.sin(np.pi * 8 / 6),
                    np.sin(np.pi * 9 / 6), np.sin(np.pi * 10 / 6), np.sin(np.pi * 11 / 6), 0] ])

# -------- выведем изображение фрактала ---------
import matplotlib.pyplot as plt
""" параметры графиков """
plt.style.use('ggplot')
plt.rcParams['font.fantasy'] = 'Arial', 'Times New Roman', 'Tahoma', 'Comic Sans MS', 'Courier'
plt.rcParams['font.family'] = 'fantasy'
linestyles = ['-', '--', '-.', ':']

# --------- выведем несколько первых точек ----------
def plot_fractal(data, n = 0, fcolor = 'green'):
    if(n == 0): n = data.shape[1]
    plt.figure()
    plt.plot(data[0,:n], data[1,:n], color= fcolor, linestyle='-')

def add_plot_fractal(data, n = 0, fcolor = 'green'):
    if(n == 0): n = data.shape[1]
    plt.plot(data[0,:n], data[1,:n], color= fcolor, linestyle=':')

"""
--------- посмотрим на графики начальных фигур ----------- 
plot_fractal(treug, fcolor='red')
plot_fractal(kust, fcolor='red')
plot_fractal(line0, fcolor='red')
plot_fractal(poligon6, fcolor='red')
plot_fractal(anglek, fcolor='red')
plot_fractal(circle, fcolor='red')
"""

"""
Для построения фракталов в качестве сжимающих отображений будем использовать следующие аффинные отображения
Аффинное отображение y = A*x +b формируется в виде numpy матрицы Ab  
"""
# 1. ZerkalReflect
def ZerkXR(alfa):
    return np.array([[alfa, 0, 0], [0, -alfa, 0]])

def ZerkYR(alfa):
    return np.array([[-alfa, 0, 0], [0, alfa, 0]])

def ZerkPlaneR(normV, M0):
    nnV = normV / np.sqrt(sum(normV * normV))
    M = np.reshape(nnV, (len(nnV),1)) * np.reshape(nnV, (1,len(nnV)))
    A = np.identity(2) - 2* M
    b = 2*nnV.dot(M0)* np.reshape(nnV, (len(nnV),1))
    Ab = np.hstack((A,b))
    return Ab

# 2. CentrReflect
def CentrOR(alfa, sx, sy):
    return np.array([[alfa, 0, sx], [0, alfa, sy]])

# 3. AngleReflect
def ShiftRotateR(alfa, fi, sx, sy):
    return np.array([[alfa*np.cos(fi), -alfa*np.sin(fi), sx], [alfa*np.sin(fi), alfa*np.cos(fi), sy]])

# 4. PlaneSqueeze
def SqueezeR(alfa1, alfa2):
    return np.array([[alfa1, 0, 0], [0, alfa2, 0]])

# 5. Shift
def ShiftR(sx, sy):
    return np.array([[0, 0, sx], [0, 0, sy]])

# Запрограммируем применение систем сжимающих итерационных функций со сгущением (ССИФ):
# ------------ функции применения SSIF к множеству точек ---------
# функция применения АП, описываемого матрицей Ab к точке M
def afftrans(Ab, M):
    M1 = np.array([M[0], M[1], 1])  # добавляем 1 к координатам точки
    return Ab.dot(M1)

# функция применения АП, описываемого матрицей Ab
# к множеству точек Xset0 в виде np.array[[x0, x1,...], [y0,y1,...]]
# на выходе формируется аналогичный массив точек
def ATransform(Ab, Xset0):
    ones = np.ones(shape=(1, Xset0.shape[1]))
    Xset1 = np.vstack((Xset0, ones))
    return np.dot(Ab, Xset1)

# функция применения списка АП  ATlist = [Ab1, Ab2, ...]
# к множеству точек Xset0 в виде np.array[[x0,x1,...], [y0, y1,...]]
# на выходе формируется аналогичный массив точек Xset1
def TSSIF(ATlist, Xset0):
    res = Xset0
    for Ab in ATlist:
        Xset1 = ATransform(Ab, Xset0)
        res = np.hstack((res, Xset1))
    return res

# ---------------------------  Иллюстрация аффинных преобразований: ------------------------

# пример применения зеркального отображения точки относительно прямой, описываемой M0, normV
normV = np.array([1,-1])
M0 = np.array([0,0])
M1 = np.array([2,1])
Ab = ZerkPlaneR(normV, M0)
afftrans(Ab, M1)

Rtreug = ATransform(Ab, treug)
plot_fractal(treug, fcolor='red')
add_plot_fractal(Rtreug, fcolor='blue')


"""
-------------- Конструируем фрактал с использованием начальных фигур и ССИФ: -------------
"""
Xset0 = line0
ATlist = [ShiftRotateR(0.618, np.pi/6, 0, 1/3), ShiftRotateR(0.618, -np.pi/6, 0, 2/3)]

Xset1 = TSSIF(ATlist, Xset0)
#plot_fractal(Xset1, fcolor='red')

Xset2 = TSSIF(ATlist, Xset1)
#plot_fractal(Xset2, fcolor='red')

niters = 7
Xset = Xset0
for i in range(niters):
    Xset = TSSIF(ATlist, Xset)

#plot_fractal(Xset, fcolor='red')
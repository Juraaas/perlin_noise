
import itertools
import random
import matplotlib.pyplot as plt
import math
import numpy as np
# from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d, RectBivariateSpline, griddata
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter


def generacja(seed):
    random.seed(seed)
    x = [random.random() for _ in range(51)]
    return x

punkty = generacja(1)
x = list(range(-21, 21))
punkty_cycle = itertools.cycle(punkty)
data = [next(punkty_cycle, punkty[0]) for i in range(42)]

#plt.plot(x, data)
#plt.show()


def inter(w_l, w_p, miejsce):
    wyjscie = w_p * miejsce + w_l*(1 - miejsce)
    return wyjscie

# print(inter(0.13436424411240122, 0.8474337369372327, 0.5))


def inter2(miejsce):
    random.seed(1)
    x = [random.random() for _ in range(21)]
    klas = []
    for i in range(len(x) - 1):
        w_l = x[i]
        w_p = x[i + 1]
        wyjscie = w_p * miejsce + w_l * (1 - miejsce)
        klas.append(wyjscie)
        print(wyjscie)
    w = [i for i in range(21)]
    z = [i + miejsce for i in w[:-1]]
    lista = x
    # plt.plot(w, lista, marker='o', markerfacecolor='blue', markersize=12)
    # plt.plot(z, klas, marker='o', markerfacecolor='orange', linestyle='None', markersize=3)
    # plt.show()
    return wyjscie

# print(inter2(0.4))


def infinite_random(seed):
    random.seed(seed)
    values = [random.random() for _ in range(10)]
    while True:
        for value in values:
            yield value


values = []
generator = infinite_random(1)
for _ in range(100):
    value = next(generator)
    values.append(value)
# plt.plot(values)
# plt.show()


def inter_nieliniowa(miejsce_przedzial):
    def przeksztalcenie(miejsce):
        return (math.cos(math.pi + miejsce * math.pi) / 2 + 0.5)

    random.seed(1)
    x = [random.random() for _ in range(21)]
    klas = []
    for i in range(len(x) - 1):
        w_l = x[i]
        w_p = x[i + 1]
        for j in range(19):
            miejsce_przedzial = j / 19
            wyjscie = w_p * miejsce_przedzial + w_l * (1 - miejsce_przedzial)
            wyjscie = przeksztalcenie(wyjscie)
            klas.append(wyjscie)

    w = [i for i in range(380)]
    lista = klas
    # plt.plot(w, lista, marker=None, markerfacecolor='blue', markersize=2)
    # plt.show()

# inter_nieliniowa(0.5)


def inter_sav(miejsce_przedzial):
    def przeksztalcenie(miejsce):
        return math.cos(math.pi + miejsce * math.pi) / 2 + 0.5

    random.seed(1)
    x = [random.random() for _ in range(21)]
    klas = []
    for i in range(len(x) - 1):
        w_l = x[i]
        w_p = x[i + 1]
        for j in range(19):
            miejsce_przedzial = j / 19
            wyjscie = w_p * miejsce_przedzial + w_l * (1 - miejsce_przedzial)
            wyjscie = przeksztalcenie(wyjscie)
            klas.append(wyjscie)

    # Zastosowanie filtra Savitzky-Golay
    lista = savgol_filter(klas, window_length=31, polyorder=2, mode='nearest')
    x = [i for i in range(len(lista))]

    x_left = [i - len(x) for i in range(-21, 0)]
    x_right = [i + len(x) for i in range(1, 22)]
    x = x_left + x + x_right
    lista = lista[-21:] + lista + lista[:21]

    # plt.plot(x, lista, marker=None, markerfacecolor='blue', markersize=2)
    # plt.show()


# inter_sav(0.5)


def inter_nieliniowa_wielomian(miejsce_przedzial):
    def wy(x):
        return 6*x**5 - 15*x**4 + 10*x**3

    random.seed(1)
    x = [random.random() for _ in range(21)]
    miejsce_przedzial = wy(miejsce_przedzial)
    klas = []
    for i in range(len(x) - 1):
        w_l = x[i]
        w_p = x[i + 1]
        wyjscie = w_p * miejsce_przedzial + w_l * (1 - miejsce_przedzial)
        klas.append(wyjscie)
    w = [i for i in range(21)]
    z = [i + miejsce_przedzial for i in w[:-1]]
    lista = x
    return wyjscie
    plt.plot(w, lista)
    plt.plot(z, klas, marker='o', markerfacecolor='None', linestyle='None', markersize=3)
    plt.show()
    print(inter_nieliniowa_wielomian(0.5))

def inter2_bot(miejsce, n):
    random.seed(1)
    x = [random.random() for _ in range(20)]
    x_extended = x + x[::-1][1:]  # rozszerzona lista z powtórzeniami

    klas = []
    for i in range(len(x_extended) - 1):
        w_l = x_extended[i]
        w_p = x_extended[i + 1]
        wyjscie = w_p * miejsce + w_l * (1 - miejsce)
        klas.append(wyjscie)

    start_idx = max(0, len(klas) - n)
    klas = klas[start_idx:start_idx + n]

    w = [i for i in range(len(klas))]
    z = [i + miejsce for i in w[:-1]]

    # plt.plot(w, klas, marker='o', markerfacecolor='blue', markersize=12)
    # plt.plot(z, klas[:-1], marker='o', markerfacecolor='orange', linestyle='None', markersize=3)
    # plt.show()
# print(inter2_bot(0.4, 200))


def perm_podw(los):
    x = [random.random() for _ in range(21)]
    n = len(x)
    y = len(x)
    if los == 1:
        n = [random.randint(0, 5) for i in range (n - 1)]
        y = [random.randint(0, 5) for i in range (y - 1)]
        for i in range(len(n)):
            n.append(y[i])
    else:
        n = [random.randint(0, 5) for i in range(n - 1)]
        y = n.copy()
        for i in range(len(n)):
            n.append(y[i])
    return n
#print(perm_podw(2))


def skoki(tab_perm, wynik_pocz, skl_0, skl_1, skl_2, skl_3):
    wynik = wynik_pocz
    wynik = tab_perm[wynik + skl_0]
    wynik = tab_perm[wynik + skl_1]
    wynik = tab_perm[wynik + skl_2]
    wynik = tab_perm[wynik + skl_3]
    return wynik

#print(funkcja_skrotu([2,0 ,4 ,1 ,3 ,2 ,0 ,4 ,1 ,3],3,1,0,1,2))


def skoki_perm(tab_perm, wynik_pocz, *skalary):
    wynik = wynik_pocz
    for s in skalary:
        wynik = tab_perm[wynik + s]
    return wynik

#print(skoki_perm([2,0 ,4 ,1 ,3 ,2 ,0 ,4 ,1 ,3],3,1,0,1))


def funkcja_skrotu_1d_perm_v0(x_c, tab_perm):
    DTW = len(tab_perm)
    skl_0 = x_c % DTW
    wynik = skoki_perm(tab_perm, 0, skl_0)
    return wynik

#print(funkcja_skrotu_1d_perm_v0(2, [2,0 ,4 ,1 ,3 ,2 ,0 ,4 ,1 ,3]))


def funkcja_skrotu_1d_perm_v0_all(x_c, tab_perm, *skalary):
    DTW = len(tab_perm)
    skl = [x % DTW for x in skalary]
    wynik = skoki_perm(tab_perm, 0, *skl)
    return wynik


def funkcja_skrotu_1d_perm_v1(tab_perm):
    DTW = len(tab_perm)
    lista = []
    for i in range(-1, 11):
        tab_perm = tab_perm + tab_perm
        skl_0 = i % DTW
        skl_1 = math.floor(i / DTW) % DTW
        wynik = skoki_perm(tab_perm, 0, skl_0, skl_1)
        lista.append(wynik)
    return lista

#print(funkcja_skrotu_1d_perm_v1([2,0 ,4 ,1 ,3]))


def funkcja_skrotu_1d_perm_v2(x_c, tab_perm):
    DTW = len(tab_perm)
    skl_0 = x_c % DTW
    skl_1 = math.floor(x_c / DTW) % DTW
    skl_2 = math.floor(x_c / (DTW * DTW)) % DTW
    wynik = skoki_perm(tab_perm, 0, skl_0, skl_1, skl_2)
    return wynik


def funkcja_skrotu_1d_perm_v3(x_c, tab_perm):
    DTW = len(tab_perm)
    tab_perm = tab_perm + tab_perm
    wyniki = []
    if x_c < 0:
        czy_ujemne = 1
        x_c = x_c * -1
    elif x_c == 0:
        czy_ujemne = 0
    else:
        czy_ujemne = 0
    wynik = tab_perm[x_c % DTW]
    x_c = math.floor(x_c / DTW)
    while x_c > 0:
        wynik = tab_perm[wynik + (x_c % DTW)]
        x_c = math.floor(x_c / DTW)
    if czy_ujemne:
        wynik = tab_perm[wynik]
    wyniki.append(wynik)
    return wyniki


wyniki = funkcja_skrotu_1d_perm_v3(1, [2, 0, 4, 1, 3])
#print(wyniki)
#plt.plot(wyniki)
#plt.show()

def interpolacja_1d_rdzen_vnlin(w_l, w_p, delta_x):
    def przeksztalcenie(miejsce):
        return (math.cos(math.pi + miejsce * math.pi) / 2 + 0.5)

    wyjscie = w_p * delta_x + w_l * (1 - delta_x)
    wyjscie = przeksztalcenie(wyjscie)
    #plt.plot([w_l, w_p], wyjscie, markerfacecolor='blue', markersize=2)
    #plt.show()

def szum_1d_pseudoperlin_vperm3_mine(x, tab_wart, tab_perm):
    x_l = x - 1
    x_p = x_l + 1
    DTW = tab_wart
    i_x_l = funkcja_skrotu_1d_perm_v3(tab_perm, x_l)
    i_x_p = funkcja_skrotu_1d_perm_v3(tab_perm, x_p)
    w_l = i_x_l
    w_p = i_x_p
    delta_x = x-x_l
    wy = inter_nieliniowa([w_l, w_p])

random.seed(1)
y = [random.random() for _ in range(21)]
tab_perm = [2, 3, 0, 4, 1, 2, 3, 0, 4, 1]
#wynik = szum_1d_pseudoperlin_vperm3(2, y, tab_perm)
#plt.plot(wynik)
#plt.show()

def szum_1d_pseudoperlin_oktawy_mine(x, tab_wart, tab_perm, oktawa_liczba, oktawa_mnoznik, oktawa_zageszczenie ):
    ampl_suma = 0
    wy = 0
    for i in range(oktawa_liczba-1):
        wys_mnoznik = oktawa_mnoznik ** i
        zageszczenie_mnoznik = oktawa_zageszczenie ** i
        ampl_suma += wys_mnoznik
        wy += wys_mnoznik
        pass

def interpolacja_wielomianowa(w_l, w_p, delta_x):
    a3 = (f(w_p) - f(w_l) - delta_x * (2 * f(w_l) + f(w_p) - 3 * f((w_l + w_p) / 2))) / ((w_p - w_l) ** 3)
    a2 = (f(w_p) - f(w_l) - a3 * (w_p - w_l) ** 3) / ((w_p - w_l) ** 2)
    a1 = (f(w_p) - f(w_l) - a3 * (w_p - w_l) ** 3 - a2 * (w_p - w_l) ** 2) / (w_p - w_l)
    a0 = f(w_l)

    wyjscie = a3 * delta_x ** 3 + a2 * delta_x ** 2 + a1 * delta_x + a0

    return wyjscie

def f(x):
    return x ** 3 - 3 * x ** 2 + 3 * x - 1

def przeksztalcenie_vkos(w_l, w_p, delta_x):
    wyjscie = w_p * math.cos(delta_x * math.pi / 2) + w_l * (1 - math.cos(delta_x * math.pi / 2))
    return wyjscie

w_l = 0.1
w_p = 0.9

x = np.linspace(w_l, w_p, 100)
y = f(x)

delta_x = np.linspace(0, 1, 100)
y_interp = [interpolacja_wielomianowa(w_l, w_p, dx) for dx in delta_x]

x_cos = np.linspace(0, 1, 100)
y_cos = [przeksztalcenie_vkos(w_l, w_p, xi) for xi in x_cos]
y_lin = [przeksztalcenie_vkos(math.cos(w_l), math.cos(w_p), dx) for dx in delta_x]

#plt.plot(delta_x, y, label='funkcja')
#plt.plot(delta_x, y_interp, label='interpolacja')
#plt.plot(delta_x, y_cos, label='cos(x)')
#plt.plot(delta_x, y_lin, label='przekształcenie(cos(x))')
#plt.legend()
#plt.show()

def szum_1d_pseudoperlin_vperm3(x, tab_wart, tab_perm):
    x_l = x - 1
    x_p = x_l + 1
    DTW = tab_wart
    #i_x_l = funkcja_skrotu_1d_perm_v3(x_l, tab_perm)
    #i_x_p = funkcja_skrotu_1d_perm_v3(x_p, tab_perm)
    #w_l = DTW[i_x_l]
    #w_p = DTW[i_x_p]
    delta_x = x-x_l
    wy = interpolacja_wielomianowa(w_l, w_p, delta_x)
    return wy

def szum_1d_pseudoperlin_oktawy(x, tab_wart, tab_perm, oktawa_liczba, oktawa_mnoznik, oktawa_zageszczenie):
    ampl_suma = 0
    wy = 0
    for okt_n in range(oktawa_liczba):
        wys_mnoznik = oktawa_mnoznik**okt_n
        zageszczenie_mnoznik = oktawa_zageszczenie**okt_n
        ampl_suma += wys_mnoznik
        w_l = math.cos((przeksztalcenie_vkos(0, 10, x) - 1) * zageszczenie_mnoznik * math.pi / 2)
        w_p = math.cos((przeksztalcenie_vkos(0, 10, x) + 1) * zageszczenie_mnoznik * math.pi / 2)
        interp_value = przeksztalcenie_vkos(w_l, w_p, 1)
        wy += wys_mnoznik * przeksztalcenie_vkos(interp_value, 0, zageszczenie_mnoznik)
    wy = wy / ampl_suma
    return wy

random.seed(1)
tab_wart = [random.random() for _ in range(10)]
tab_perm = [2, 3, 0, 4, 1, 2, 3, 0, 4, 1]

x = np.linspace(0, 10, 100)
y = [szum_1d_pseudoperlin_oktawy(przeksztalcenie_vkos(0, 10, xi), tab_wart, tab_perm, 5, 0.5, 3) for xi in x]

#plt.plot(x, y)
#plt.show()



def oktawian(x, tab_wart, tab_perm, oktawa_liczba, oktawa_mnoznik, oktawa_zageszczenie):
    wy = 0
    for oktawa in range(oktawa_liczba):
        czestotliwosc = 2 ** oktawa
        amplituda = 1 / czestotliwosc
        wy += amplituda * np.random.randn(len(y), len(x))
    return wy

random.seed(1)
tab_wart = [random.random() for _ in range(10)]
tab_perm = [2, 3, 0, 4, 1, 2, 3, 0, 4, 1]

x = np.linspace(0, 10, 100)
y = [oktawian(x, tab_wart, tab_perm, 6, 0.5, 2) for xi in x]

#plt.plot(x, y)
#plt.show()


def funkcja_skrotu_uniw_perm_v3(tab_skl, tab_perm, DTW):
    wynik = 0
    for skl_c in tab_skl:
        if skl_c < 0:
            czy_ujemne = 1
            skl_c = -skl_c
        else:
            czy_ujemne = 0
        wynik = tab_perm[(wynik + (skl_c % DTW)) % len(tab_perm)]
        skl_c = skl_c // DTW
        while skl_c > 0:
            wynik = tab_perm[(wynik + (skl_c % DTW)) % len(tab_perm)]
            skl_c = skl_c // DTW
        if czy_ujemne:
            wynik = tab_perm[wynik]
    return wynik

def szum_2d_uniw_pseudoperlin_vperm3(x, y, tab_wart, tab_perm, DTW, oktawa_liczba, oktawa_mnoznik, oktawa_zageszczenie):
    ampl_suma = 0
    wy = 0
    for okt_n in range(oktawa_liczba):
        wys_mnoznik = oktawa_mnoznik ** okt_n
        zageszczenie_mnoznik = oktawa_zageszczenie ** okt_n
        ampl_suma += wys_mnoznik
        skl_x = x * zageszczenie_mnoznik
        skl_y = y * zageszczenie_mnoznik
        skl = [int(skl_x), int(skl_y)]
        wy += wys_mnoznik * funkcja_skrotu_uniw_perm_v3(skl, tab_perm, DTW)
    wy = wy / ampl_suma
    return wy

random.seed(1)
tab_wart = [random.random() for _ in range(20)]
tab_perm = [1, 3, 2, 0, 1, 3, 2, 0]
DTW = len(tab_wart)
OKTAWA_LICZBA = 3
OKTAWA_MNOZNIK = 0.5
OKTAWA_ZAGESZCZENIE = 3

x = np.arange(-1, 10, 1)
y = np.arange(0, 7, 1)

xx, yy = np.meshgrid(x, y)
zz = np.zeros((len(x), len(y)))

for i in range(len(x)):
    for j in range(len(y)):
        zz[i][j] = szum_2d_uniw_pseudoperlin_vperm3(x[i], y[j], tab_wart, tab_perm, DTW, OKTAWA_LICZBA, OKTAWA_MNOZNIK, OKTAWA_ZAGESZCZENIE)

zz = zz.T
#fig, ax = plt.subplots(figsize=(10,10))
#im = ax.imshow(zz, cmap='hot', interpolation='nearest')
#ax.set_title('2D Perlin Noise')
#fig.tight_layout()
#plt.show()

#fig, ax = plt.subplots(figsize=(10,10))
#ax.set_title('2D Perlin Noise')

#ax.pcolormesh(xx, yy, zz, cmap='hot')
#ax.set_aspect('equal')

#plt.show()



def funkcja_skrotu_2d(tab_skl, permu):
    wynik = 0
    DTW = len(tab_perm)
    permu = permu + permu

    for skl_c in tab_skl:
        if skl_c < 0:
            czy_ujemne = 1
            skl_c = abs(skl_c)
        else:
            czy_ujemne = 0
        wynik = permu[int(wynik + skl_c % DTW)]
        skl_c = math.floor(skl_c / DTW)
        while skl_c > 0:
            wynik = permu[int(wynik + skl_c / DTW)]
            skl_c = math.floor(skl_c / DTW)
        if czy_ujemne:
            wynik = permu[wynik]
    return wynik

#print(funkcja_skrotu_2d([-123, 45], [2, 3, 0, 4, 1]))
#print(skoki_perm([2, 3, 0, 4, 1, 2, 3, 0, 1, 4], 0, 3, 2, 1, 0, 5, 4))
#print(funkcja_skrotu_2d([0], [1, 3, 2, 0, 1, 3, 2, 0]))


def interpolacja_2d_vlin(w_ld, w_lg, w_pd, w_pg, delta_x, delta_y):
    wynik = w_ld * (1-delta_x)*(1-delta_y) + w_lg * (1-delta_x)*delta_y + \
            w_pd * delta_x*(1-delta_y) + w_pg * delta_x*delta_y
    return wynik

x_vals = y_vals = np.linspace(0, 1, 50)
x, y = np.meshgrid(x_vals, y_vals)
w_ld, w_lg, w_pd, w_pg = 0, 1, 3, 2

z = np.zeros_like(x)
for i in range(len(x)):
    for j in range(len(y)):
        z[i][j] = interpolacja_2d_vlin(w_ld, w_lg, w_pd, w_pg, x[i][j], y[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)

plt.show()


def interpolacja_2d_rdzen_vnlin(w_ld, w_lg, w_pd, w_pg, delta_x, delta_y):
    delta_x = 6 * delta_x**5 - 15 * delta_x**4 + 10 * delta_x**3
    delta_y = 6 * delta_y**5 - 15 * delta_y**4 + 10 * delta_y**3
    wynik = w_ld * (1-delta_x)*(1-delta_y) + w_lg * (1-delta_x)*delta_y + \
            w_pd * delta_x*(1-delta_y) + w_pg * delta_x*delta_y
    return wynik

x_vals = y_vals = np.linspace(0, 1, 50)
x, y = np.meshgrid(x_vals, y_vals)
z = np.zeros_like(x)

w_ld, w_lg, w_pd, w_pg = 0, 1, 3, 2

for i in range(len(x)):
    for j in range(len(y)):
        z[i][j] = interpolacja_2d_rdzen_vnlin(w_ld, w_lg, w_pd, w_pg, x[i][j], y[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
plt.show()


def interpolacja_2d_cala_vperm_nlin(x, y, tab_wart, tab_perm):
    x_l = int(x)
    x_p = x_l + 1
    x_delta = x - x_l
    y_d = int(y)
    y_g = y_d + 1
    y_delta = y - y_d
    i_ld = funkcja_skrotu_uniw_perm_v3([x_l, y_d], tab_perm,32)
    i_lg = funkcja_skrotu_uniw_perm_v3([x_l, y_g], tab_perm,32)
    i_pd = funkcja_skrotu_uniw_perm_v3([x_p, y_d], tab_perm,32)
    i_pg = funkcja_skrotu_uniw_perm_v3([x_p, y_g], tab_perm,32)
    w_ld = tab_wart[i_ld]
    w_lg = tab_wart[i_lg]
    w_pd = tab_wart[i_pd]
    w_pg = tab_wart[i_pg]
    return interpolacja_2d_rdzen_vnlin(w_ld, w_lg, w_pd, w_pg, x_delta, y_delta)


x = np.linspace(-1, 10, 10)
y = np.linspace(0, 7, 10)
X, Y = np.meshgrid(x, y)
tab_wart = np.linspace(0, 1, 5)
tab_perm = np.array([1, 3, 2, 0, 1, 3, 2, 0])

wynik = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        wynik[i, j] = interpolacja_2d_cala_vperm_nlin(X[i, j], Y[i, j], tab_wart, tab_perm)

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, projection='3d')
# ax3.plot_surface(X, Y, wynik)
# plt.show()


def szum_2d_pseudoperlin_oktawy(x, y, tab_wart, tab_perm, oktawa_liczba, oktawa_mnoznik, oktawa_zageszczenie):
    ampl_suma = 0
    wy = 0
    for okt_n in range(len(oktawa_liczba)):
        wys_mnoznik = oktawa_mnoznik ** okt_n
        zageszczenie_mnoznik = oktawa_zageszczenie ** okt_n
        ampl_suma += wys_mnoznik
        wy += wys_mnoznik * interpolacja_2d_cala_vperm_nlin(x*zageszczenie_mnoznik, y*zageszczenie_mnoznik, tab_wart, tab_perm)
    wy = wy / ampl_suma
    return wy

x = np.linspace(-1, 10, 10)
y = np.linspace(0, 7, 10)
X, Y = np.meshgrid(x, y)

tab_wart = np.linspace(0, 1, 5)
tab_perm = np.array([1, 3, 2, 0, 1, 3, 2, 0])
oktawa_liczba = [0, 1, 2, 3, 4, 5]

wynik = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        wynik[i, j] = szum_2d_pseudoperlin_oktawy(X[i, j], Y[i, j], tab_wart, tab_perm, oktawa_liczba, 0.25, 2)

# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111, projection='3d')
# ax4.plot_surface(X, Y, wynik)
# plt.show()

def generacja2(seed):
    np.random.seed(seed)
    x = np.arange(-5, 5, 0.05)
    return x

def generuj_szum():
    x = np.arange(-5, 5, 0.05)
    y = np.arange(1, 6, 0.05)
    X, Y = np.meshgrid(x, y)
    szum = np.random.rand(len(y), len(x))
    return X, Y, szum

def progowanie(szum, prog1, prog2):
    pola = np.zeros_like(szum)
    pola[np.logical_and(szum >= prog1, szum < prog2)] = 0.3
    pola[szum >= prog2] = 1
    return pola

X, Y, szum = generuj_szum()
pola = progowanie(szum, 0.4, 0.95)

# fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
# im = ax.imshow(pola, extent=[-5, 5, 1, 6], cmap='terrain', origin='lower', aspect='auto', interpolation='none')
# cbar = plt.colorbar(im, label='Typ pola')
# plt.title('Mapa szumu z progowaniem')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


def generuj_szum2():
    x = np.arange(-5, 5.05, 0.05)
    y = np.arange(1, 6.05, 0.05)
    X, Y = np.meshgrid(x, y)

    szum = np.random.uniform(0, 1, size=X.shape)
    szum = uniform_filter(szum, size=5)

    return X, Y, szum


X, Y, szum = generuj_szum2()

# fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.imshow(szum, extent=[-5, 5, 1, 6], cmap='viridis', origin='lower', aspect='auto', interpolation='bilinear')
# cbar = plt.colorbar(im, label='Wartość')
# plt.title('Mapa szumu z klastrami')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


def generuj_szum4():
    x = np.arange(-5, 5.05, 0.05)
    y = np.arange(1, 6.05, 0.05)
    X, Y = np.meshgrid(x, y)

    szum = np.random.random(size=X.shape)
    prog_1 = 0.3
    prog_2 = 0.9
    szum = np.where(szum < prog_1, 0.1, szum)
    szum = np.where(np.logical_and(szum >= prog_1, szum < prog_2), 0.5, szum)
    szum = np.where(szum >= prog_2, 0.9, szum)

    return X, Y, szum


X, Y, szum = generuj_szum4()

# fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.imshow(szum, extent=[-5, 5, 1, 6], cmap='viridis', origin='lower', aspect='auto', interpolation='none')
# cbar = plt.colorbar(im, ticks=[0.1, 0.5, 0.9], label='Prog')
# cbar.ax.set_yticklabels(['0..0.3', '0.3..0.9', '0.9..1'])
# plt.title('Mapa szumu z plamami')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


def generuj_szum5():
    x = np.arange(-5, 5.05, 0.05)
    y = np.arange(1, 6.05, 0.05)
    X, Y = np.meshgrid(x, y)

    # Generowanie szumu na podstawie funkcji falowej
    szum = np.sin(X) * np.sin(Y)

    prog_1 = 0.3
    prog_2 = 0.9
    szum = np.where(szum < prog_1, 0.1, szum)
    szum = np.where(np.logical_and(szum >= prog_1, szum < prog_2), 0.5, szum)
    szum = np.where(szum >= prog_2, 0.9, szum)

    return X, Y, szum


X, Y, szum = generuj_szum5()

# fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.imshow(szum, extent=[-5, 5, 1, 6], cmap='viridis', origin='lower', aspect='auto', interpolation='none')
# cbar = plt.colorbar(im, ticks=[0.1, 0.5, 0.9], label='Prog')
# cbar.ax.set_yticklabels(['0..0.3', '0.3..0.9', '0.9..1'])
# plt.title('Mapa szumu z plamami')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


def generuj_szum6(szum_skala):
    x = np.arange(-5, 5, 0.05)
    y = np.arange(1, 6, 0.05)
    X, Y = np.meshgrid(x, y)

    szum = np.zeros_like(X)
    oktawy = 3
    for oktawa in range(oktawy):
        czestotliwosc = 2 ** oktawa
        amplituda = 1 / czestotliwosc
        szum += amplituda * np.random.randn(len(y), len(x))

    # Normalizacja szumu do przedziału [0, 1]
    szum -= np.min(szum)
    szum /= np.max(szum)

    # Tworzenie "wysp"
    prog = 0.55
    szum = np.where(szum < prog, 0, szum)

    # Skalowanie szumu
    szum *= szum_skala

    return X, Y, szum

X, Y, szum = generuj_szum6(szum_skala=1)

# fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.imshow(szum, extent=[-5, 5, 1, 6], cmap='terrain', origin='lower', aspect='auto', interpolation='bilinear')
# cbar = plt.colorbar(im, label='Wysokość')
# plt.title('Mapa wysp')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

def generuj_szum7(szum_skala):
    x = np.arange(-5, 5, 0.05)
    y = np.arange(1, 6, 0.05)
    X, Y = np.meshgrid(x, y)

    # Generowanie szumu Perlina
    szum = np.zeros_like(X)
    oktawy = 6
    for oktawa in range(oktawy):
        czestotliwosc = 2 ** oktawa
        amplituda = 1 / czestotliwosc
        szum += amplituda * np.random.randn(len(y), len(x))

    # Normalizacja szumu do przedziału [0, 1]
    szum -= np.min(szum)
    szum /= np.max(szum)

    # Tworzenie "wysp"
    prog = 0.1
    szum = np.where(szum < prog, 0, szum)

    # Skalowanie szumu
    szum *= szum_skala

    return X, Y, szum

def interpolacja_wielomianowa(x, y, z, xi, yi):
    points = np.column_stack((x, y))
    values = z.flatten()
    xi_flat = xi.flatten()
    yi_flat = yi.flatten()
    points_flat = np.column_stack((xi_flat, yi_flat))

    wynik = griddata(points, values, points_flat, method='cubic')
    wynik = wynik.reshape(xi.shape)
    wynik = wynik-1

    return wynik

X, Y, szum = generuj_szum7(szum_skala=3)

xi = np.linspace(-5, 5, 500)
yi = np.linspace(1, 6, 500)
XI, YI = np.meshgrid(xi, yi)

wynik = interpolacja_wielomianowa(X.flatten(), Y.flatten(), szum.flatten(), XI, YI)

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
im = ax.imshow(wynik, extent=[-5, 5, 1, 6], cmap='terrain', origin='lower', aspect='auto')
cbar = plt.colorbar(im, label='Wysokość')
plt.title('Mapa wysp')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def generuj_szum_perlin_1d(dlugosc, oktawa, lacznik):
    szum = np.zeros(dlugosc)
    czestotliwosc = 2 ** oktawa
    amplituda = 1 / czestotliwosc
    szum_dla_oktawy = amplituda * np.random.randn(dlugosc // czestotliwosc)
    szum_dla_oktawy = np.resize(szum_dla_oktawy, dlugosc)  # Dopasowanie rozmiaru tablicy
    szum += szum_dla_oktawy
    return szum

dlugosc = 100
oktawy = [1, 3, 4]
lacznik = 0.25

szumy = []
for oktawa in oktawy:
    szum = generuj_szum_perlin_1d(dlugosc, oktawa, lacznik)
    szumy.append(szum)

fig, ax = plt.subplots()
for i, szum in enumerate(szumy):
    x = np.arange(len(szum))
    ax.plot(x, szum, label=f"Oktawa {oktawy[i]}")

# ax.legend()
# plt.xlabel('Indeks')
# plt.ylabel('Wartość')
# plt.title('Szum Perlin 1D')
# plt.show()

w_ld = 0

x_vals = y_vals = np.linspace(0, 1, 50)
x, y = np.meshgrid(x_vals, y_vals)

z = np.full_like(x, w_ld)  # Użyj funkcji full_like, aby wypełnić tablicę z wartością w_ld

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)

plt.show()
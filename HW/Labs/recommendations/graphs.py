import matplotlib.pyplot as plt

# Шрифты, их размеры и размеры графиков
plt.rcParams["figure.figsize"] = 11, 7
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 8
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 16

# Построение графика
vals = ...
fig, ax = plt.subplots()
# 2 графика на одной картинке fig, axs = plt.subplots(2)
ax.plot(vals, label='Test label')
ax.set_yscale('log') # логарифмическая шкала по оси y
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title('Graph title')
ax.grid()
ax.legend()

plt.tight_layout() # убрать наложение текста друг на друга


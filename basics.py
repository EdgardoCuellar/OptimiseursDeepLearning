import matplotlib.pyplot as plt
import numpy as np

OPTIMIZE_NUMBER = 5
OPTIMIZER_STR = ["Stochastic Gradient Descent", "Mini Batch", "SGD avec momentum", "Adaptive Gradient Descent", "RMSProp"]
OPTIMIZER_COLORS = ["#D50000", "#2962FF", "#FFD600", "#00C853", "#3E2723"]

def plotAll(x, y):
    for i in range(0, OPTIMIZE_NUMBER):
      plt.plot(x, y[i], color=OPTIMIZER_COLORS[i])
    plt.legend(OPTIMIZER_STR, loc='upper right')
    plt.xlabel('Nombre d\'epoche')
    plt.ylabel('Temps en secondes')
    plt.title("Temps des différents algorithmes d'optimisation, 10.000 données")
    plt.show()

def main():
    times = []

    if False:
        times = [[310.4433925151825, 73.6064465045929, 78.74599123001099, 75.32900953292847, 74.58012843132019],
                [1410.9272332191467, 325.5799369812012, 319.5950469970703, 323.4821608066559, 307.00171422958374],
                [3374.1272325515747, 677.444916009903, 685.7470302581787, 690.5538585186005, 692.8289818763733]]
    else: 
        times = [[58.834877729415894, 11.966078281402588, 11.716942071914673, 12.2535879611969, 11.628226041793823],
                [228.38012170791626, 47.80546689033508, 50.01236081123352, 52.6740403175354, 53.74828386306763],
                [562.7100427150726, 124.2383201122284, 120.02731442451477, 120.82826137542725, 119.95251798629761]]

    times_T = np.array(times).T
    times = times_T.tolist()

    x_epoch = [2, 10, 25]

    plotAll(x_epoch, times)

if __name__ == '__main__':
    main()
    
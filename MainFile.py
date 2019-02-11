import pickle
import numpy as np
import matplotlib.pyplot as plt
from progressbar import *
from ModeFunctions import mode_functions_class
import Extrapolation
import LMSUsingTF
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import bokeh

mfc = mode_functions_class()


def SNR_mode(d, n, k, total_runs):
    noise_amp_vec = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 100]
    SNR = np.zeros((len(noise_amp_vec), 1))
    MMSE_neuron = np.zeros((len(noise_amp_vec), 1))
    MMSE_PI = np.zeros((len(noise_amp_vec), 1))
    widgets = ['Processing estimation SNR mode: ', Percentage(), ' ', Bar()]
    max_val = len(noise_amp_vec) * total_runs
    bar = ProgressBar(widgets=widgets, maxval=int(max_val)).start()
    LMSUsingTF.LMS_layers()
    for i in range(len(noise_amp_vec)):
        SNR[i], MMSE_neuron[i], MMSE_PI[i] = LMSUsingTF.LMS_layers(d, n, k, noise_amp=noise_amp_vec[i],
                                                                   total_runs=total_runs)
        bar.update(i * total_runs)
    bar.finish()
    ind_sorted = np.argsort(SNR[:, 0])
    SNR = SNR[ind_sorted]
    MMSE_neuron = MMSE_neuron[ind_sorted]
    MMSE_PI = MMSE_PI[ind_sorted]

    print('Plotting')
    output_file("MSE_SNR.html")
    p = figure(title='Mean MSE as function of SNR')
    p.grid.grid_line_alpha = 0.3
    p.xaxis.axis_label = 'SNR'
    p.yaxis.axis_label = 'Mean MSE'
    p.line(SNR[:, 0], MMSE_neuron[:, 0], legend='LMS using TF', color='blue')
    p.line(SNR[:, 0], MMSE_PI[:, 0], legend='Pseudo Inverse', color='Orange')
    p.legend.location = "top_left"
    show(p)


def main():
    # plt.interactive(True)
    # x dimensions
    d = 1000
    # number of samples
    n = 100
    # The effective dimension
    k = 10
    # number of runs
    total_runs = 100
    noise_amp_all = 4
    batch_size_all = 5

    SNR_mode(d, n, k, total_runs)


if __name__ == "__main__":
    main()

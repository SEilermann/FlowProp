from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_sample_training(truth,predictions,path,sample_nr=0,title='Title'):
    """
    Plot a sample of the training data during the training process
    """

    truth = truth[sample_nr].cpu().detach().numpy()
    predictions = predictions[sample_nr].cpu().detach().numpy()

    plt.plot(truth, label='Ground Truth')
    plt.plot(predictions, label='Predictions')
    plt.xlabel('Distance')
    plt.ylabel('Water Level')
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

def calc_cd(h0, q=0.05, w=0.5, g=9.81, p=0.3):
    v = q / (w * h0)
    ht = (h0-p)
    Ht = (v**2)/(2*g) + ht
    cd = (3 * q) / (2 * w * (2*g)**(1/2) * Ht**(3/2))
    return cd

def calc_area_diff(y_true, y_pred, x_range=(0, 2.5, 501)):
    x = np.linspace(x_range[0], x_range[1], x_range[2])
    area_true = np.abs(simpson(y_true, x=x))
    area_pred = np.abs(simpson(y_pred, x=x))
    return np.abs(area_pred - area_true)

def plot_area_diff(y_true, y_pred, file_path="", x_range=(0, 2.5, 501)):
    plt.figure(figsize=(20, 4))
    x = np.linspace(x_range[0], x_range[1], x_range[2])
    plt.plot(x, y_true, label='Simulated Waterline')
    plt.plot(x, y_pred, label='Predicted Waterline')
    plt.fill_between(x, y_true, y_pred, where=(y_pred > y_true), interpolate=True, color='red', alpha=0.3, label='Areal Deviation [$m^2$]')
    plt.fill_between(x, y_true, y_pred, where=(y_pred <= y_true), interpolate=True, color='red', alpha=0.3)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.xlim(x_range[0], x_range[1])
    plt.savefig(file_path)
    plt.close()

def add_gaussian(x, device="cuda", mean=0, std=0.05):
    gaussian_noise = np.random.normal(mean, std, x.shape)
    x = np.array(x.cpu())
    x = x + gaussian_noise
    return torch.tensor(x, dtype=torch.float32).to(device)



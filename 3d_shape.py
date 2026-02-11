import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ShapePlotter:
    def __init__(self, sigma_t, sigma_lambda, t0=0, lambda0=0, unit='nm'):
        self.sigma_t = sigma_t
        self.sigma_lambda = sigma_lambda
        self.t0 = t0
        self.lambda0 = lambda0
        self.unit = unit

    def _gaussian_2d(self, t, lambda_):
        return np.exp(-((t - self.t0)**2 / (self.sigma_t**2) + (lambda_ - self.lambda0)**2 / (self.sigma_lambda**2)))

    def _cubic_3d(self, t, lambda_):
        t = np.asarray(t)
        lambda_ = np.asarray(lambda_)
        
        z = np.zeros(np.broadcast(t, lambda_).shape)
        mask = (np.abs(t - self.t0) <= self.sigma_t) & (np.abs(lambda_ - self.lambda0) <= self.sigma_lambda)
        z[mask] = 1
        return z

    def plot_3d(self, shape='gaussian'):
        t = np.linspace(self.t0 - 2.5 * self.sigma_t, self.t0 + 2.5 * self.sigma_t, 100)
        lambda_ = np.linspace(self.lambda0 - 2.5 * self.sigma_lambda, self.lambda0 + 2.5 * self.sigma_lambda, 100)
        T, LAMBDA = np.meshgrid(t, lambda_)

        if shape == 'gaussian':
            Z = self._gaussian_2d(T, LAMBDA)
            title = f'3D Gaussian (σ_t={self.sigma_t}, σ_λ={self.sigma_lambda})'
        elif shape == 'cubic':
            Z = self._cubic_3d(T, LAMBDA)
            title = f'3D Cubic (σ_t={self.sigma_t}, σ_λ={self.sigma_lambda})'
        else:
            raise ValueError("Shape must be 'gaussian' or 'cubic'")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T, LAMBDA, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.set_xlabel('Time (ps)')
        if self.unit == 'nm':
            ylabel = f'Wavelength ({self.unit})'
        elif self.unit == 'cm-1':
            ylabel = f'Wavenumber ({self.unit})'
        else:
            ylabel = f'Spectral Unit ({self.unit})' # Fallback for unknown units
        ax.set_ylabel(ylabel)
        ax.set_zlabel('Intensity')
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5)
        plt.tight_layout()
        plt.show()

    def plot_projections(self, shape='gaussian'):
        t = np.linspace(self.t0 - 2.5 * self.sigma_t, self.t0 + 2.5 * self.sigma_t, 100)
        lambda_ = np.linspace(self.lambda0 - 2.5 * self.sigma_lambda, self.lambda0 + 2.5 * self.sigma_lambda, 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if shape == 'gaussian':
            z_t = self._gaussian_2d(t, self.lambda0)
            z_lambda = self._gaussian_2d(self.t0, lambda_)
        elif shape == 'cubic':
            z_t = self._cubic_3d(t, self.lambda0)
            z_lambda = self._cubic_3d(self.t0, lambda_)
        else:
            raise ValueError("Shape must be 'gaussian' or 'cubic'")

        # t-axis projection
        ax1.plot(t, z_t, 'b-', linewidth=2)
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Field Intensity')
        ax1.set_title(f't-axis projection')
        ax1.grid(True, alpha=0.3)

        # lambda-axis projection
        ax2.plot(lambda_, z_lambda, 'r-', linewidth=2)
        if self.unit == 'nm':
            xlabel_lambda = f'Wavelength ({self.unit})'
        elif self.unit == 'cm-1':
            xlabel_lambda = f'Wavenumber ({self.unit})'
        else:
            xlabel_lambda = f'Spectral Unit ({self.unit})' # Fallback for unknown units
        ax2.set_xlabel(xlabel_lambda)
        ax2.set_ylabel('Intensity')
        ax2.set_title(f'λ-axis projection')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage at 805 nm
    sigma_t_805 = 46.85e-3  # 1/2 Width at 1/e of the field gaussian along t-axis, temporal axis
    sigma_lambda = 5.1674 # Width along λ-axis, spectral axis
    lambda0 = 803.1577

    # Gaussian plot in nm
    plotter_nm = ShapePlotter(sigma_t_805, sigma_lambda, lambda0=lambda0, unit='nm')
    plotter_nm.plot_3d('gaussian')
    plotter_nm.plot_projections('gaussian')

    # Cubic plot in nm around 805 nm
    plotter_nm.plot_3d('cubic')
    plotter_nm.plot_projections('cubic')


    # Example usage at 1045 nm
    sigma_t_1045 = 107.5e-3  # Width along t-axis, temporal axis
    sigma_lambda = 3.7891 # Width along λ-axis, spectral axis
    lambda0 = 1040.8500

    # Gaussian plot in nm
    plotter_nm = ShapePlotter(sigma_t_1045, sigma_lambda, lambda0=lambda0, unit='nm')
    plotter_nm.plot_3d('gaussian')
    plotter_nm.plot_projections('gaussian')

    # Cubic plot in nm around 1045 nm
    plotter_nm.plot_3d('cubic')
    plotter_nm.plot_projections('cubic')


    # Example with wavenumbers around 805 nm
    lambda0_wn = 12451.7349
    sigma_lambda_wn = 81.4550
    plotter_wn = ShapePlotter(sigma_t_805, sigma_lambda_wn, lambda0=lambda0_wn, unit='cm-1')
    plotter_wn.plot_3d('gaussian')
    plotter_wn.plot_projections('gaussian')

    # Example with wavenumbers around 1045 nm
    lambda0_wn = 9607.7701
    sigma_lambda_wn = 36.1284
    plotter_wn = ShapePlotter(sigma_t_1045, sigma_lambda_wn, lambda0=lambda0_wn, unit='cm-1')
    plotter_wn.plot_3d('gaussian')
    plotter_wn.plot_projections('gaussian')

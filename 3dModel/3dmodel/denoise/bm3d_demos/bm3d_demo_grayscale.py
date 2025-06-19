"""
Grayscale BM3D denoising demo file, based on
Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354.
"""


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bm3d import bm3d, BM3DProfile
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
from skimage.restoration import estimate_sigma


def denosie(imagename,visual = False):
    # Experiment specifications
    # imagename = './2024on/2024On_new_20.png'
    # imagename = 'cameraman256.png'
    
    # 如果输入是npz格式的文件，则用以下方式打开
    y = imagename
    
    noise_type = 'gw'
    noise_var = 0.0298  # Noise variance
    seed = 0  # seed for pseudorandom noise realization

    noise_sigma = estimate_sigma(z.squeeze(2))
    # 显示图片估计出噪声的水平
    print("Estimated noise var = {}".format(noise_sigma**2))
    # Generate noise with given PSD
    noise, psd, kernel = get_experiment_noise(noise_type, noise_sigma**2, seed, y.shape)
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD

    # z = np.atleast_3d(y) + np.atleast_3d(noise)

    z = np.atleast_3d(y)


    # Call BM3D With the default settings.
    y_est = bm3d(z, psd)

    # To include refiltering:
    # y_est = bm3d(z, psd, 'refilter')

    # For other settings, use BM3DProfile.
    # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm3d(z, psd, profile);

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm3d(z, sqrt(noise_var));

    psnr = get_psnr(y, y_est)
    print("PSNR:", psnr)

    # PSNR ignoring 16-pixel wide borders (as used in the paper), due to refiltering potentially leaving artifacts
    # on the pixels near the boundary of the image when noise is not circulant
    psnr_cropped = get_cropped_psnr(y, y_est, [16, 16])
    print("PSNR cropped:", psnr_cropped)

    # Ignore values outside range for display (or plt gives an error for multichannel input)
    y_est = np.minimum(np.maximum(y_est, 0), 1)
    z_rang = np.minimum(np.maximum(z, 0), 1)

    if visual == True:
        plt.title("y, z, y_est")
        plt.imshow(np.concatenate((y, np.squeeze(z_rang), y_est), axis=1), cmap='gray')
        plt.show()
    
    return y_est


if __name__ == '__main__':
    imagename = './2024on/2024On_new_20.png'
    denosie(imagename,visual = True)


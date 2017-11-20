import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials import mnist

from denoising_autoencoder import corruption, training

ENCODING_LENGTH = 2000
BATCH_SIZE = 64
CORRUPTION_PROB = 0.6
LEARNING_RATE = 5e-4
NUM_STEPS = 20000

DIM = 28
PIXELS = DIM**2


def _imsave(filename, X, title, n_width=10, n_height=10):
    for i in range(n_height * n_width):
        if i < len(X):
            img = X[i]
            ax = plt.subplot(n_height, n_width, i + 1)
            for d in ('bottom', 'top', 'left', 'right'):
                ax.spines[d].set_linewidth(2.)
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            plt.tick_params(
                axis='both', which='both', bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off', labelright='off'
            )
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.savefig(filename)


def _reshape_vec_to_im(vec):
    return np.reshape(vec, (-1, DIM, DIM))


def _plot(original, corrupted, reconstructed, weights):
    original_imgs = _reshape_vec_to_im(original)
    corrupted_imgs = _reshape_vec_to_im(corrupted)
    reconstruction_imgs = _reshape_vec_to_im(reconstructed)
    normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
    weights_imgs = _reshape_vec_to_im(normalized_weights.T)
    _imsave('../img/mnist/original.png', original_imgs[:100, :, :], 'original')
    _imsave('../img/mnist/corrupted.png', corrupted_imgs[:100, :, :], 'corrupted')
    _imsave('../img/mnist/reconstruction.png', reconstruction_imgs[:100, :, :], 'reconstruction')
    _imsave('../img/mnist/weights.png', weights_imgs[:100, :, :], 'weights')


def main():
    data_sets = mnist.input_data.read_data_sets('../data/')
    corrupter = corruption.MaskingCorrupter(CORRUPTION_PROB)
    corrupted, reconstructed, weights = training.fit_and_evaluate(
        data_sets.train, data_sets.test.images, ENCODING_LENGTH, BATCH_SIZE, corrupter,
        LEARNING_RATE, NUM_STEPS, PIXELS
    )
    _plot(data_sets.test.images, corrupted, reconstructed, weights)


if __name__ == '__main__':
    main()

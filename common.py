import numpy as np


def salt_and_pepper(img, prob, shape):
    """salt and pepper noise for mnist"""
    rnd = np.random.rand(shape)
    noisy = img[:]
    noisy[rnd < prob/2] = 0.
    noisy[rnd > 1 - prob/2] = 1.
    return noisy

def normalize(imgs):
    return imgs / 255.

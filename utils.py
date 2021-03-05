import numpy as np
from scipy.interpolate import interp1d



def add_noise(coords, scale=0.00):
    """
    adds gaussian noise to strokes
    """
    coords = np.copy(coords)
    coords += np.random.normal(loc=0.0, scale=scale, size=coords.shape)
    return coords


def interpolate(MAX_STROKE_LEN, stroke):
    """
    interpolates strokes using cubic spline
    """
    coords = np.zeros([18, MAX_STROKE_LEN], dtype=np.float32)

    if len(stroke) > 3:
        for j in range(18):

            f_x = interp1d(np.arange(len(stroke)), stroke[:, j], kind='cubic')
            xx = np.linspace(0, len(stroke) - 1, MAX_STROKE_LEN)
            x_new = f_x(xx)

            coords[j, :] = x_new
    coords = np.transpose(coords)
    return coords



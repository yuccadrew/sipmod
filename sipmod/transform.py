from typing import Tuple, Dict, Optional, Union

import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d
from empymod.utils import check_time
from empymod.model import tem


# https://empymod.emsig.xyz/en/stable/manual/transforms.html
# https://empymod.emsig.xyz/en/stable/gallery/tdomain/step_and_impulse.html
class FourierDLF:
    time: ndarray
    freq: ndarray
    ft: str
    ftarg: Dict

    def __init__(self, time: ndarray, verbose: int = 2):
        self.time, self.freq, self.ft, self.ftarg = type(self).argft(time)

    @staticmethod
    def argft(freqtime: ndarray,
              signal: int = 1,
              ft: str = 'dlf',
              ftarg: Dict = {},
              verbose: int = 2) -> Tuple[ndarray, ndarray, str, Dict]:
        if ft == 'dlf' and not ftarg:
            ftarg = {'dlf': 'key_81_CosSin_2009'}
        if signal is not None:
            out = check_time(freqtime, signal, ft, ftarg, verbose)
        else:
            out = (None, freqtime.copy(), ft, ftarg)
        return out

    @staticmethod
    def interpolate(x: ndarray,
                    y: ndarray,
                    xnew: ndarray,
                    axis: int = None) -> ndarray:
        if axis is None:  # interpolate 1d array
            assert y.ndim == 1, "dimension of y is not 1"
            fill_value = (y[0], y[-1])
            ynew = interp1d(x, y,
                            bounds_error=False,
                            fill_value=fill_value)(xnew)
        elif axis == 0:  # interpolate 2d array along the first axis
            assert y.ndim == 2, "dimension of y is not 2"
            ynew = np.zeros((xnew.shape[0], y.shape[1]), dtype=y.dtype)
            for i in range(y.shape[1]):
                fill_value = (y[0, i], y[-1, i])
                ynew[:, i] = interp1d(x, y[:, i],
                                      bounds_error=False,
                                      fill_value=fill_value)(xnew)
        elif axis == 1:  # interpolate 2d array along the second axis
            assert y.ndim == 2, "dimension of y is not 2"
            ynew = np.zeros((y.shape[0], xnew.shape[0]), dtype=y.dtype)
            for i in range(y.shape[0]):
                fill_value = (y[i, 0], y[i, -1])
                ynew[i, :] = interp1d(x, y[i, :],
                                      bounds_error=False,
                                      fill_value=fill_value)(xnew)
        else:
            raise NotImplementedError
        return ynew

    def tospectral(self, tEM: ndarray):
        raise NotImplementedError

    def totemporal(self,
                   fEM: ndarray,
                   dcEM: Optional[Union[ndarray, float]] = None,
                   time: Optional[ndarray] = None) -> ndarray:
        fEM = fEM.reshape(self.freq.shape[0], -1)
        off = np.empty(fEM.shape[1])
        signal = 1
        if dcEM is None:
            # charge-up case: sine-transform is enforced; real part matters
            tEM, conv = tem(fEM,
                            off,
                            self.freq,
                            self.time,
                            signal,
                            self.ft,
                            self.ftarg)
        else:
            # charge-down case: instead of setting signal as -1 and using
            # the cosine-transform, we do the DC-computation and subtraction
            # manually.
            # See empymod Gallery/Time Domain/Step and impulse responses
            # and look for the following part.
            # For switch-off to work properly you need empymod-version bigger
            # than 1.3.0! You can do it with previous releases too, but you
            # will have to do the DC-computation and subtraction manually, as
            # is done here for ee_xx_step.
            # exDC = ee_xx_step(res[1], aniso[1], rec[0], 60*60)
            # ex = exDC - ee_xx_step(res[1], aniso[1], rec[0], t)
            if dcEM is not ndarray:
                tEM, conv = tem(dcEM-fEM,
                                off,
                                self.freq,
                                self.time,
                                signal,
                                self.ft,
                                self.ftarg)
            else:
                tEM, conv = tem(np.einsum('ij,j->ij',
                                          np.ones(fEM.shape),
                                          dcEM)-fEM,  # to be improved
                                off,
                                self.freq,
                                self.time,
                                signal,
                                self.ft,
                                self.ftarg)
        if time is not None:
            return np.squeeze(type(self).interpolate(self.time,
                                                     tEM,
                                                     time,
                                                     axis=0))
        else:
            return np.squeeze(tEM)

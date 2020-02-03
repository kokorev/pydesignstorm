#!/usr/bin/env python
# coding=utf-8
"""
Tools for adjusting the ERA5 distribution to observational data
"""
from scipy import stats
import numpy as np
from functools import lru_cache
import pandas as pd
import netCDF4
from .geohelpers import grid_cells_in_poly


class BiasCorrector:
    """
    Class takes "true" data, "biased" data, and name of a distribution
    Adjust biased data using following procedure
    The distribution is fitted for both data sets then
    quantile for biased values is taken from the biased distribution, the value of this quantile from true distribution replaces the original value
    D_b(V_b) -> Q -> D_t(Q) -> V_corrected
    BiasCorrector object implement __getitem__ function that takes slice of biased_vals and return corrected data
    Data that was not in initial input can be corrected using functions .corr_f(value) or vectorized version .corr_f_vec(arr)
    """

    def __init__(self, true_vals, biased_vals, distribution='pearson3', min_val=0, use_fast_correct=True):
        self.ta = self.to_numpy(true_vals)
        self.ba = self.to_numpy(biased_vals)
        self.no_rain_lvl = min_val
        self.is_corrected_mask = np.zeros(self.ba.shape, dtype=bool)
        self._ca = self.ba.copy()
        self.ta_flat = self.ta.flatten()
        self.ta_flat = self.ta_flat[np.logical_not(np.isnan(self.ta_flat))]
        self.ba_flat = self.ba.flatten()
        self.correct_zero_percent()
        self.distr_class = getattr(stats, distribution)

        self.true_distr = self.distr_class(
            *self.distr_class.fit(self.ta_flat[self.ta_flat > self.z_value])
        )
        self.biased_distr = self.distr_class(
            *self.distr_class.fit(self.ba_flat[self.ba_flat > self.z_value])
        )
        self.use_fast_corr = use_fast_correct
        if self.use_fast_corr:
            self._fast_corr_f = self.get_fast_corr_f()
        self.corr_f_vec = self._fast_corr_f if self.use_fast_corr else np.vectorize(self.slow_corr_f)

    @staticmethod
    def to_numpy(var):
        if isinstance(var, pd.DataFrame):
            res_var = var.to_numpy()
        elif isinstance(var, (np.ndarray, np.generic)):
            res_var = var
        else:
            res_var = np.array(var)
        return res_var

    def corr_f(self, v):
        if self.use_fast_corr:
            res = self.fast_corr_f(v)
        else:
            res = self.slow_corr_f(v)
        return res

    @lru_cache(maxsize=None)
    def slow_corr_f(self, v):
        q = self.biased_distr.cdf(v)
        q = q if q < 1 else 1 - 1 * 10 ** -16
        return self.true_distr.ppf(q)

    def get_fast_corr_f(self):
        quants = [1 * 10 ** -16] + [v / 100 for v in range(1, 100)] + [1 - (1 * 10 ** -16)]
        true = [self.true_distr.ppf(q) for q in quants]
        biased = [self.biased_distr.ppf(q) for q in quants]
        slope, intercept, r_value, p_value, std_err = stats.linregress(true, biased)
        if r_value > 0.99:
            fy = lambda y: (y - intercept) / slope
        else:
            # todo: use logger
            print('r_value for fast correction linear regression is below 0.99. Falling back to slow correction. R-value = %f' % r_value)
            self.use_fast_corr = False
            raise AssertionError
        return fy

    def fast_corr_f(self, v):
        try:
            res = self._fast_corr_f(v)
        except TypeError:
            try:
                self._fast_corr_f = self.get_fast_corr_f()
            except AssertionError:
                res = self.slow_corr_f(v)
            else:
                res = self._fast_corr_f(v)
        return res

    @property
    def _ca_flat(self):
        return self._ca.flatten()

    @staticmethod
    def _zero_percent(vals):
        n = vals.shape[0]
        n_zero = n - np.count_nonzero(vals)
        return n_zero / (n / 100)

    @staticmethod
    def _less_then_percent(vals, th):
        n = vals.shape[0]
        n_below = vals[vals <= th].shape[0]
        return n_below / (n/100)

    @property
    def true_zero_percent(self):
        return self._zero_percent(self.ta_flat)

    @property
    def biased_zero_percent(self):
        return self._zero_percent(self.ba_flat)

    def correct_zero_percent(self):
        """
        calculates percent of zero values in observations, set equal amount of bottom values to zero in corrected values
        """
        self.z_value = np.quantile(self.ba_flat, self._less_then_percent(self.ta_flat, self.no_rain_lvl) / 100)
        below_z_mask = self.ba <= self.z_value
        self._ca[below_z_mask] = 0
        self.is_corrected_mask[below_z_mask] = True

    def __getitem__(self, item):
        """
        return bias corrected values
        correcting done on the first request of particular value
        """
        corrected_mask = self.is_corrected_mask[item]
        if not np.all(corrected_mask):
            mask = np.logical_not(corrected_mask)
            self._ca[mask] = self.corr_f_vec(self._ca[mask])
            self.is_corrected_mask[mask] = True
        return self._ca[item]

    def get_coefs(self, item):
        """
        return coefficients = corrected / biased

        :param item: slice passed to __getitem__
        :return:
        """
        coefs = self[item] / self.ba[item]
        return coefs


class BiasCorrectorHourly(BiasCorrector):
    """

    """

    def __init__(self, true_vals_daily, biased_vals_hourly, distribution='pearson3', min_val=0):
        assert isinstance(biased_vals_hourly, pd.DataFrame)
        self.ba_hourly = biased_vals_hourly
        ba_daily = self.ba_hourly.resample('d').sum()
        self.ba_daily_index = ba_daily.index
        super().__init__(true_vals_daily, ba_daily, distribution=distribution, min_val=min_val)

    def __getitem__(self, item):
        if isinstance(item, slice):
            item = (item, slice(None, None, None))
        ca = super().__getitem__(item)
        coefs = ca / self.ba[item]
        coefs[np.isnan(coefs)] = 0.
        ti = self.ba_daily_index[item[0]]
        ti_hourly = pd.date_range(ti.min(), ti.max() + pd.DateOffset(hours=23, minutes=59, seconds=59), freq=self.ba_hourly.index.inferred_freq)
        ti_hourly = ti_hourly.intersection(self.ba_hourly.index)
        ci = self.ba_hourly.columns[item[1]]
        if not hasattr(ci, '__iter__'):
            ci = [ci]
        coefs_daily = pd.DataFrame(coefs, index=ti, columns=ci)
        coefs_hourly = coefs_daily.reindex(index=ti_hourly, method='ffill')
        ca_hourly = self.ba_hourly.loc[coefs_hourly.index, ci] * coefs_hourly
        return ca_hourly


class ERA5InShapeCorrector:

    def __init__(self, nc_fn, shp_fn, obs_data, coors, nc_var='tp', distr_name='pearson3'):
        self.nc_var = nc_var
        self.distr_name = distr_name
        self.ds = netCDF4.Dataset(nc_fn)
        self.var = np.array(self.ds.variables[self.nc_var]) * 1000
        self.lats = self.ds.variables['latitude'][:]
        self.lons = self.ds.variables['longitude'][:]
        self.time = self.ds.variables['time']
        self.dti = pd.DatetimeIndex(netCDF4.num2date(self.ds.variables['time'][:], self.time.units))
        self.grid_cell_latlon_per_poly = grid_cells_in_poly(self.lats, self.lons, shp_fn)


#!/usr/bin/env python
# coding=utf-8
"""

"""
import numpy as np
import pandas as pd
import netCDF4
from functools import lru_cache
import itertools
from PIL import Image

from .common import get_rp_f, get_top_event_from_daily, get_event_dates_from_daily_ts, get_events_average
from .geohelpers import grid_cells_in_poly, stations_in_poly, closest_station_map, stations_closest_to_poly
from .bias_corrector import BiasCorrectorHourly


class ERA5Reader:
    """
    Class provides basic functionality of reading ERA5 data and getting return period functions
    """

    def __init__(self, nc_fn, nc_var='tp', lat_var='latitude', lon_var='longitude'):
        """
        :param nc_fn: path to netCDF file
        :param nc_var: name of the variable containing precipitation data in netCDF file
        :param lat_var: name of latitude axis in netCDF file
        :param lon_var: name of longitude axis in netCDF file
        """
        self.nc_var = nc_var
        self.ds = netCDF4.Dataset(nc_fn)
        self.var = np.array(self.ds.variables[self.nc_var]) * 1000
        self.lats = self.ds.variables[lat_var][:]
        self.lons = self.ds.variables[lon_var][:]
        self.time = self.ds.variables['time']
        self.dti = pd.DatetimeIndex(netCDF4.num2date(self.ds.variables['time'][:], self.time.units))

    def get_era_cell(self, ilat, ilon):
        """
        Return data in selected grid cell as pandas Series
        :param ilat: Lat index of a grid cell (not lat value!)
        :param ilat: Lon index of a grid cell (not Lon value!)
        """
        return pd.Series(self.var[:, ilat, ilon], index=self.dti)

    def get_rp_f_cell(self, ilat, ilon):
        """
        for selected cell return function that takes return period as an input and return value corresponding to this return period
        :param ilat: cell latitude index
        :param ilon: cell longitude index
        :return: return period function
        """
        era = self.get_era_cell(ilat, ilon)
        era = era.resample('d').sum()
        rp_f = get_rp_f(era.to_numpy().flatten(), steps_in_year=365)
        return rp_f


class ERA5ReaderShape(ERA5Reader):
    """ Provide same functional as ERA5Reader but per shape
    """

    def __init__(self, nc_fn, shp_fn, nc_var='tp', lat_var='latitude', lon_var='longitude'):
        """
        :param nc_fn: path to netCDF file
        :param shp_fn: path to .shp file in Lat/Lon coordinates
        :param nc_var: name of the variable containing precipitation data in netCDF file
        :param lat_var: name of latitude axis in netCDF file
        :param lon_var: name of longitude axis in netCDF file
        """
        super().__init__(nc_fn, nc_var=nc_var, lat_var=lat_var, lon_var=lon_var)
        lat_step = abs(self.lats[0] - self.lats[1])
        lon_step = abs(self.lats[1] - self.lats[0])
        self.grid_cell_latlon_per_poly = grid_cells_in_poly(self.lats - lat_step / 2, self.lons + lon_step / 2, shp_fn)
        self.grid_cell_latlon_per_poly = {k: v for k, v in self.grid_cell_latlon_per_poly.items() if len(v) > 0}
        self.shp_names = list(self.grid_cell_latlon_per_poly.keys())
        self.rp_f = {key: None for key in self.shp_names}

    @lru_cache(maxsize=None)
    def get_era_in_shape(self, shape_name):
        """
        Return DataFrame with all the data in selected shape
        :param shape_name: name of the shape as specified in provided shape file
        :return: pandas.DataFrame
        """
        inds = np.array(self.grid_cell_latlon_per_poly[shape_name])
        lst = [self.get_era_cell(ilat, ilon) for ilat, ilon in zip(inds[:, 0], inds[:, 1])]
        reg_dat = pd.DataFrame(lst).transpose()
        return reg_dat

    def get_rp_f_in_shape(self, shape_name):
        """
        get return period function for selected shape
        :param shape_name: name of the shape as specified in provided shape file
        :return: function
        """
        if self.rp_f[shape_name] is None:
            era = self.get_era_in_shape(shape_name)
            era = era.resample('d').sum()
            era = era.mean(axis=1)
            rp_f = get_rp_f(era.to_numpy(), steps_in_year=365)
            self.rp_f[shape_name] = rp_f
        return self.rp_f[shape_name]


class ERA5CorrectorShape(ERA5ReaderShape):
    """
    Provide same functional as ERA5ReaderShape but data is bias corrected first
    """

    def __init__(self, nc_fn, shp_fn, observed_data, coors=None, distr_name='pearson3', empty_shapes='drop',
                 nc_var='tp', lat_var='latitude', lon_var='longitude'):
        """
        :param nc_fn: path to netCDF file
        :param shp_fn: path to .shp file in Lat/Lon coordinates
        :param observed_data: station data as pandas.DataFrame with pandas.DatetimeIndex and columns representing individual stations
        :param coors: list of coordinates [[lat1,lon1], [lat2, lon2], ...] where first element correspond to first column in the observed_data.
        If None then all of the station considered to be in every shape
        :param distr_name: distribution to use in bias correction procedure. Accept any distribution name from scipy.stats
        :param empty_shapes: (drop | nearest). What to do if no station is in a shape. Drop - drop this shape. Nearest - use nearest station
        :param nc_var: name of the variable containing precipitation data in netCDF file
        :param lat_var: name of latitude axis in netCDF file
        :param lon_var: name of longitude axis in netCDF file
        """
        super().__init__(nc_fn, shp_fn, nc_var=nc_var, lat_var=lat_var, lon_var=lon_var)
        self.distr_name = distr_name
        self.obs = observed_data
        if coors is not None:
            self.st_ind_per_poly = stations_in_poly(coors, shp_fn)
            if empty_shapes == 'drop':
                self.st_ind_per_poly = {k: v for k, v in self.st_ind_per_poly.items() if len(v) > 0}
                self.shp_names = list(set(self.grid_cell_latlon_per_poly.keys()) & set(self.st_ind_per_poly.keys()))
            elif empty_shapes == 'nearest':
                empty_names = [k for k, v in self.st_ind_per_poly.items() if len(v) == 0]
                closest = stations_closest_to_poly(coors, shp_fn)
                for shp_name in empty_names:
                    self.st_ind_per_poly[shp_name] = closest[shp_name]
            else:
                raise KeyError("Incorrect value of empty_shapes")
        else:
            self.st_ind_per_poly = {key: self.obs.columns for key in self.shp_names}

        self.bc_obj = {key: None for key in self.shp_names}

    @lru_cache(maxsize=None)
    def get_obs_in_shape(self, shape_name):
        return self.obs[self.st_ind_per_poly[shape_name]]

    def get_bc_obj_in_shape(self, shape_name):
        if self.bc_obj[shape_name] is None:
            obs = self.get_obs_in_shape(shape_name)
            era = super().get_era_in_shape(shape_name)
            bc = BiasCorrectorHourly(obs, era, distribution=self.distr_name)
            self.bc_obj[shape_name] = bc
        return self.bc_obj[shape_name]

    @lru_cache(maxsize=None)
    def get_era_in_shape(self, shape_name):
        """
        Return DataFrame with all the data in selected shape. Data is biased corrected.
        :param shape_name: name of the shape as specified in provided shape file
        :return: pandas.DataFrame
        """
        bc_obj = self.get_bc_obj_in_shape(shape_name)
        return bc_obj[:, :]


class ERA5CorrectedCells(ERA5Reader):
    """
    Provide access to bias corrected data
    Bias correction done per cell distribution adjusted to fit that of a nearest station
    """

    def __init__(self, nc_fn, observed_data, coors, distr_name='pearson3', nc_var='tp', lat_var='latitude', lon_var='longitude'):
        super().__init__(nc_fn, nc_var=nc_var, lat_var=lat_var, lon_var=lon_var)
        self.distr_name = distr_name
        self.obs = observed_data
        self.closest_st_map = closest_station_map(coors, self.lats, self.lons)
        self.n_lats = self.lats.shape[0]
        self.n_lons = self.lons.shape[0]
        self.bc_obj = {k: None for k in self.obs.columns}

    def get_obs_in_cell(self, ilat, ilon):
        st_ind = self.closest_st_map[ilat, ilon]
        obs = self.obs[st_ind]
        return obs

    def get_bc_obj_in_cell(self, ilat, ilon):
        st_ind = self.closest_st_map[ilat, ilon]
        if self.bc_obj[st_ind] is None:
            obs = self.get_obs_in_cell(ilat, ilon)
            era_inds = np.argwhere(self.closest_st_map == st_ind)
            lst = [pd.Series(self.var[:, ilat, ilon], index=self.dti) for ilat, ilon in era_inds]
            era = pd.DataFrame(lst).transpose()
            bc = BiasCorrectorHourly(pd.DataFrame(obs), era, distribution=self.distr_name)
            self.bc_obj[st_ind] = bc
        else:
            bc = self.bc_obj[st_ind]
        return bc

    @lru_cache(maxsize=None)
    def get_era_cell(self, ilat, ilon):
        """
        Return bias corrected data in selected grid cell as pandas Series
        :param ilat: Lat index of a grid cell (not lat value!)
        :param ilat: Lon index of a grid cell (not Lon value!)
        """
        st_ind = self.closest_st_map[ilat, ilon]
        inds = np.argwhere(self.closest_st_map == st_ind)
        col_ind = np.argwhere((inds == [ilat, ilon]).all(axis=1))[0][0]
        bc = self.get_bc_obj_in_cell(ilat, ilon)
        return bc[:, col_ind]

    def get_cell_ind_iterator(self):
        return itertools.product(np.arange(self.n_lats), np.arange(self.n_lons))


class StormDesignerCorrectedERA5(ERA5CorrectorShape, ERA5CorrectedCells):
    """
    high level class for design storm functions using per shape bias correction
    """
    days_before = 10
    days_after = 5

    def get_event_scaled(self, shape_name, rp):
        """
        Return design storm calculated by scaling maximum observed historical events to the intensity of the given return period
        :param shape_name:
        :param rp: return period in years
        :return:
        """
        era = self.get_era_in_shape(shape_name)
        top_event = get_top_event_from_daily(pd.DataFrame(era, index=self.dti))
        event_daily_layer = pd.Series(top_event.mean(axis=1)).rolling(24).sum().max()
        rp_f = self.get_rp_f_in_shape(shape_name)
        coef = rp_f(rp) / event_daily_layer
        return top_event * coef

    def get_event_mean(self, shape_name, rp):
        """
        Return design storm event calculated by averaging historical events with intensity from RP - 1 year to RP + 1 year
        :param shape_name:
        :param rp: return period in years
        :return: pandas.DataFrame
        """
        rp_f = self.get_rp_f_in_shape(shape_name)
        val_min, val_max = rp_f(rp - 1), rp_f(rp + 1)
        era = self.get_era_in_shape(shape_name)
        era.resample('d').sum()
        dates = get_event_dates_from_daily_ts(era, val_min, val_max)
        design_storm = get_events_average(era, dates, self.days_before, self.days_after)
        return design_storm

    def get_rp_map_shape(self, rp):
        """
        return map of return period values where return periods calculated per shape
        :param rp: return period in years
        :return:
        """
        rp_map = np.empty((self.var.shape[1], self.var.shape[2]))
        rp_map[:, :] = np.nan
        for shape_name in self.grid_cell_latlon_per_poly:
            inds = np.array(self.grid_cell_latlon_per_poly[shape_name])
            lat_inds, lon_inds = inds[:, 0], inds[:, 1]
            rp_f = self.get_rp_f_in_shape(shape_name)
            rp_val = rp_f(rp)
            rp_map[lat_inds, lon_inds] = rp_val
        return rp_map

    def get_rp_map_cell(self, rp):
        """
        return map of return period values calculated per cell independently
        :param rp:
        :return:
        """
        rp_map = np.empty((self.var.shape[1], self.var.shape[2]))
        rp_map[:, :] = np.nan
        for ilat, ilon in self.get_cell_ind_iterator():
            try:
                rp_f = self.get_rp_f_cell(ilat, ilon)
                rp_val = rp_f(rp)
            except TypeError:
                rp_val = -1
            rp_map[ilat, ilon] = rp_val
        return rp_map

    def get_scaled_max_map(self, rp):

        rp_map = np.empty((self.var.shape[1], self.var.shape[2]))
        rp_map[:, :] = np.nan
        for shape_name in self.grid_cell_latlon_per_poly:
            e = self.get_event_scaled(shape_name, rp)
            rol_24h = pd.DataFrame(e).rolling(24).sum().dropna()
            inds = np.array(self.grid_cell_latlon_per_poly[shape_name])
            lat_inds, lon_inds = inds[:, 0], inds[:, 1]
            max_ind = rol_24h.mean(axis=1).idxmax()
            rp_map[lat_inds, lon_inds] = rol_24h.loc[max_ind, :]
        return rp_map

    def save_event(self, event, fn, coors_header=True, shape_name=None, coors_delim=';', **kw):
        """
        Dump event data in txt file
        :param event: event array from .get_event_* function
        :param fn: name of the resulting file
        :param coors_header: Use cell coordinates as column names. Overwrites header key word (default True)
        :param shape_name: Name of the shape to get coordinates from
        :param coors_delim: separator to use between lat lon coordinates in the header
        :param kw: all additional arguments passed to numpy.savetxt()
        :return:
        """
        if coors_header:
            if 'delimiter' in kw:
                delim = kw['delimiter']
            else:
                delim = ' '
            coors = np.array([(self.lats[v[0]], self.lons[v[1]]) for v in self.grid_cell_latlon_per_poly[shape_name]])
            cols = ['{}{}{}'.format(c[0], coors_delim, c[1]) for c in coors]
            hd = delim.join(cols)
            kw['header'] = hd
        np.savetxt(fn, event, **kw)

    @staticmethod
    def save_map_tif(m, fn):
        """
        Save map as tif. Resulting map is format used by openLISEM
        :param m: map as returned from get_*_map()
        :param fn: result file name
        :return:
        """
        img = Image.fromarray(m)
        img.save(fn)


class PerCellDesignerCorrectedERA5(ERA5CorrectedCells):
    def get_rp_map(self, rp):
        """
        return map of return period values calculated per cell independently
        :param rp:
        :return:
        """
        rp_map = np.empty((self.var.shape[1], self.var.shape[2]))
        rp_map[:, :] = np.nan
        for ilat, ilon in self.get_cell_ind_iterator():
            try:
                rp_f = self.get_rp_f_cell(ilat, ilon)
                rp_val = rp_f(rp)
            except TypeError:
                rp_val = -1
            rp_map[ilat, ilon] = rp_val
        return rp_map

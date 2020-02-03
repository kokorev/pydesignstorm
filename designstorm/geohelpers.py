#!/usr/bin/env python
# coding=utf-8
"""
Functions to detect points stations and grid cells inside shape
"""
import shapefile as shp
from shapely.geometry import Polygon, Point
import itertools
import numpy as np


def convert_lon_to_positive(lon):
    """ Convert longitude to 181+ format """
    if lon < 0:
        lon = 360 + lon
    return lon


def read_shp(shp_fn):
    res = {}
    sf = shp.Reader(shp_fn)
    for name, sp in zip(sf.records(), sf.shapes()):
        lonmin, latmin, lonmax, latmax = sp.bbox
        lonmin, lonmax = convert_lon_to_positive(lonmin), convert_lon_to_positive(lonmax)
        box = (lonmin, latmin, lonmax, latmax)
        if lonmin < 0 or lonmax < 0:
            polygon_points = [[convert_lon_to_positive(cors[0]), cors[1]] for cors in sp.points]
        else:
            polygon_points = sp.points
        poly = Polygon(polygon_points)
        res[name[0]] = [poly, box]
    return res


def grid_cells_in_poly(lats, lons, shp_fn):
    """
    Return list of grid cells inside the given shape
    :param lats: list of grid cell latitudes
    :param lons: list of grid cell longitudes
    :param shpfile: path to *.shp file in lat/lon coordinates
    :return: dictionary of the following format
        {shape_name_1: [[lat_ind_1, lon_ind_1], [lat_ind_2, lon_ind_2], ...],
        shape_name_2: [...]}
        """
    res = dict()
    shapes_dict = read_shp(shp_fn)
    for name, (poly, box) in shapes_dict.items():
        res_tmp = []
        lonmin, latmin, lonmax, latmax = box
        for ilat, lat in enumerate(lats):
            for ilon, lon in enumerate(lons):
                lon = convert_lon_to_positive(lon)
                if not (lonmin <= lon <= lonmax and latmin <= lat <= latmax):
                    continue
                pnt = Point(lon, lat)
                if poly.contains(pnt):
                    res_tmp.append([ilat, ilon])
        res[name] = res_tmp
    return res


def stations_in_poly(coors, shp_fn):
    """
    Return list of grid cells inside the given shape
    :param lats: list of grid cell latitudes
    :param lons: list of grid cell longitudes
    :param shpfile: path to *.shp file in lat/lon coordinates
    :return: dictionary of the following format
        {shape_name_1: [[lat_ind_1, lon_ind_1], [lat_ind_2, lon_ind_2], ...],
        shape_name_2: [...]}
        """
    res = dict()
    shapes_dict = read_shp(shp_fn)
    points = [Point(lon, lat) for lat, lon in coors]
    coors = [(lat, convert_lon_to_positive(lon)) for lat, lon in coors]
    for name, (poly, box) in shapes_dict.items():
        res_tmp = []
        lonmin, latmin, lonmax, latmax = box
        for ind, (lat, lon) in enumerate(coors):
            if not (lonmin <= lon <= lonmax and latmin <= lat <= latmax):
                continue
            pnt = points[ind]
            if poly.contains(pnt):
                res_tmp.append(ind)
        res[name] = res_tmp
    return res


def stations_closest_to_poly(coors, shp_fn):
    res = dict()
    shapes_dict = read_shp(shp_fn)
    coors = [(lat, convert_lon_to_positive(lon)) for lat, lon in coors]
    points = [Point(lon, lat) for lat, lon in coors]
    for name, (poly, box) in shapes_dict.items():
        res_tmp = []
        for ind, (lat, lon) in enumerate(coors):
            pnt = points[ind]
            dist = poly.distance(pnt)
            res_tmp.append(dist)
        res_tmp = np.array(res_tmp)
        res[name] = [res_tmp.argmin()]
    return res


def closest_station_map(st_coors, grid_lats, grid_lons):
    n_lats = grid_lats.shape[0]
    n_lons = grid_lons.shape[0]
    closest_st_map = np.empty((n_lats, n_lons))
    st_points = [Point(lon, lat) for lat, lon in st_coors]
    for ilat, ilon in itertools.product(np.arange(n_lats), np.arange(n_lons)):
        lat = grid_lats[ilat]
        lon = grid_lons[ilon]
        gridcell = Point(lon, lat)
        min_ind = np.array([st.distance(gridcell) for st in st_points]).argmin()
        closest_st_map[ilat, ilon] = min_ind
    return closest_st_map

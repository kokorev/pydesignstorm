#!/usr/bin/env python
# coding=utf-8
"""
This is a code for downloading ERA5 data
Reburies CDS account and CDS API installed. See instructions on https://cds.climate.copernicus.eu/api-how-to
"""
import cdsapi


class ERA5Downloader:
    """ This class is simplifies downloading of hourly precipitation data
    Example -
    >>> d = ERA5Downloader('era_download_test.nc')
    >>> d.set_latlon_box(62, 28, 58, 32)
    >>> d.set_years([2018, 2019])
    >>> d.run()
    Request might take a long time to complete, you can monitor the status on https://cds.climate.copernicus.eu/cdsapp#!/yourrequests
    """
    dataset = 'reanalysis-era5-single-levels'
    q = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'grid': '0.25/0.25',
        'time': ['%02i:00' % i for i in range(0, 24)],
        'day': ["%02i" % i for i in range(1, 32)],
        'month': ["%02i" % i for i in range(1, 13)],
        'year': ['2019']
    }

    def __init__(self, fn, var='total_precipitation'):
        self.cds = cdsapi.Client()
        self.filename = fn
        self.q['variable'] = var
        self.variable = var

    def set_latlon_box(self, top_lat, left_lon, bottom_lat, right_lon):
        self.q['area'] = "{:5.2f}/{:5.2f}/{:5.2f}/{:5.2f}".format(top_lat, left_lon, bottom_lat, right_lon)

    def set_years(self, years):
        self.q['year'] = ['%i' % y for y in years]

    def run(self):
        print("live request can be see on https://cds.climate.copernicus.eu/cdsapp#!/yourrequests")
        self.cds.retrieve(self.dataset, self.q, self.filename)

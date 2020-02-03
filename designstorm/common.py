import numpy as np
import pandas as pd
from datetime import timedelta


def get_event_dates_from_daily_ts(ts, val_min, val_max):
    """
    Get list of dates of the events close to the provided quantile
    :param ts: data as pandas series
    :param val_min: minimum rainfall for an event
    :param val_max: maximum rainfall for an event
    :return:
    """
    mask = (ts > val_min) & (ts < val_max)
    dates = ts.index[mask]
    return dates


def get_events_df(ts, dates, days_before, days_after):
    pieces = [ts[d - timedelta(days=days_before):d + timedelta(days=days_after)] for d in dates]
    pieces = [p.to_numpy() for p in pieces if not np.any(np.isnan(p))]
    return pieces


def get_events_average(ts, dates, days_before, days_after):
    pieces = [ts[d - timedelta(days=days_before):d + timedelta(days=days_after)] for d in dates]
    pieces = [p for p in pieces if not np.any(np.isnan(p))]
    mean_len = np.array([p.shape[0] for p in pieces]).mean()
    res = np.array([p.to_numpy() for p in pieces if p.shape[0] >= mean_len])
    return res.mean(axis=0)


def get_rp_f(pr, steps_in_year=365):
    pr = np.sort(pr)
    n = pr.shape[0]
    m = np.arange(n)[::-1] + 1
    rps = (n+1)/m
    y_returns = rps / steps_in_year
    c = np.polyfit(pr[y_returns > 1], np.log(y_returns[y_returns > 1]), deg=1)
    fy = lambda y: (np.log(y) - c[1]) / c[0]
    return fy


def split_ds_on_index_gap(ds, index_delta):
    """
    ds - pandas series
    index_delta - if difference btween steps is large then given values series is splited
    return list of pandas series splited on a gap in the index
    """
    df = pd.DataFrame(ds)
    df['index'] = df.index
    df['diff'] = df['index'].diff() > timedelta(hours=1)
    df['group_n'] = df['diff'].cumsum()
    return [v[1][0] for v in df.groupby('group_n')]


def get_top_event_from_daily(d_hourly, q=0.99, days_before=10, days_after=5):
    reg_dat_roll = d_hourly.rolling(24).sum().dropna()
    reg_dat_roll_mean = reg_dat_roll.mean(axis=1)
    val_min = np.quantile(reg_dat_roll_mean.values.flatten(), 0.99)
    val_max = reg_dat_roll_mean.max()
    dates = get_event_dates_from_daily_ts(reg_dat_roll_mean, val_min, val_max)
    vals = [d_hourly.loc[d - timedelta(days=1):d, :].mean(axis=1) for d in dates]
    all_events_ds = pd.concat(vals, axis=0)
    events = split_ds_on_index_gap(all_events_ds, timedelta(hours=1))
    mean = get_events_average(d_hourly, [e.idxmax() for e in events], days_before, days_after)
    return mean


def get_hourly_scale_from_top_event(d_hourly, target_rp, q=0.99, days_before=10, days_after=5):
    top_event = get_top_event_from_daily(d_hourly)
    event_daily_layer = pd.Series(top_event.mean(axis=1)).rolling(24).sum().max()
    reg_dat_roll = d_hourly.rolling(24).sum().dropna()
    reg_dat_roll_mean = reg_dat_roll.mean(axis=1)
    rp_f = get_rp_f(reg_dat_roll_mean.to_numpy(), steps_in_year=365)
    coef = rp_f(500) / event_daily_layer
    return top_event * coef

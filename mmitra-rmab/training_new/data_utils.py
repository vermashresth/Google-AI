import logging
import glob
import os

from datetime import date

import numpy as np
import pandas as pd


def date_to_month(date_int: int):
    """
    Convert date(int format) to find month (1-12)
    """
    return date.fromordinal(date(2018, 1, 1).toordinal() + date_int).month


def calls_so_far(user_id: int, date_int: int, df, threshold: int = 30):
    """
    Find call attempts, connections, engagements
    """
    calls = df[(df["user_id"] == user_id) & (df["startdate"] <= date_int)]
    attempts = calls.shape[0]
    calls = calls[calls["Technical Status"] == 1]
    successful_attempts = calls.shape[0]
    calls = calls[calls["duration"] > 0]
    connections = calls.shape[0]
    calls = calls[calls["duration"] >= threshold]
    engagements = calls.shape[0]

    return attempts, successful_attempts, connections, engagements


def pad_sequences(x, n=None, dim=None):
    if n:
        padded_x = np.zeros((len(x), n, dim), dtype=np.float32)
    else:
        padded_x = np.zeros((len(x), max([len(k) for k in x]), dim), dtype=np.float32)

    for ctr, k in enumerate(x):
        if n:
            if padded_x.shape[2] != 1:
                padded_x[ctr, : min(n, k.shape[0]), :] = k[-min(n, k.shape[0]) :]
            else:
                padded_x[ctr, : min(n, k.shape[0]), :] = k[
                    -min(n, k.shape[0]) :
                ].reshape(min(n, k.shape[0]), 1)

        else:
            if padded_x.shape[2] != 1:
                padded_x[ctr, : len(k), :] = k
            else:
                padded_x[ctr, : len(k), :] = k.reshape(len(k), 1)

    return padded_x


def get_csv_files(data_dir: str):
    return glob.glob(os.path.join(data_dir, "*.csv"))


def date_to_int(date: str, fmt: str = "%Y-%m-%d") -> int:
    """
    Convert string date to int using formula 'month+(year-2018)*12'
    """
    try:
        date = pd.to_datetime(date, format=fmt) - pd.to_datetime(
            "2018-01-01", format="%Y-%m-%d"
        )
        return int(date.days)
    except:
        if type(date) == pd.Series:
            return pd.NaT

        logging.warning(f"Can't convert {date} using format {fmt}. Outputting NA.")
        return pd.NaT


def time_to_int(time: str, fmt: str) -> int:
    """
    Convert string date to hour
    """
    try:
        date = pd.to_datetime(time, format=fmt)
        return date.hour
    except:
        if type(date) == pd.Series:
            return pd.NaT

        logging.warning(f"Can't convert {time} using format {fmt}. Outputting NA.")
        return pd.NaT


def wrap_date_to_int(fmt: str):
    return lambda x: date_to_int(x, fmt)


def wrap_time_to_int(fmt: str):
    return lambda x: time_to_int(x, fmt)

def enroll_delivery_status_to_int(inp: str):
    return 1 if inp == 'D' else 0

def education_to_int(inp: int) -> int:
    """
    Educaton level to int. If 7(illiterate) convert to 0
    """
    try:
        return int(inp)#0 if int(inp) == 7 else int(inp)
    except:
        logging.warning(f"{inp} is not a valid education. Outputting NA.")
        return pd.NaT


def phowner_to_int(inp: str) -> int:
    """
    Convert phone_owner to int
    """
    vals = {"woman": 0, "husband": 1, "family": 2, 'women': 0}
    if inp.lower().strip() in vals.keys():
        return vals[inp.lower().strip()]
    else:
        #logging.warning(f"{inp} is not a valid phone owner. Outputting NA.")
        return 2 #pd.NaT


def ch_type_to_int(inp: str) -> int:
    """
    Convert ChannelType to int
    """
    vals = {"community": 0, "hospital": 1, "armman": 2}
    if inp.lower().strip() in vals.keys():
        return vals[inp.lower().strip()]
    else:
        logging.warning(f"{inp} is not a valid channel type. Outputting NA.")
        return pd.NaT


def income_bracket_to_int(inp: str) -> int:
    """
    Convert income_bracket to int
    """
    return int(inp) - 1
    vals = {
        "0-5000": 0,
        "5000-10000": 1,
        "10000-15000": 2,
        "15000-20000": 3,
        "20000-25000": 4,
        "25000-30000": 5,
        "30000 and above": 6,
    }
    if inp.lower().strip() in vals.keys():
        return vals[inp.lower().strip()]
    else:
        f"{inp} is not a valid income bracket. Outputting NA."
        return pd.NaT


def age_to_int(age: int) -> int:
    """
    Convert age to int
    """
    try:
        age = int(age)
        assert 13 <= age <= 60
        return age
    except:
        logging.warning(f"{age} is not a valid age. Outputting NA.")
        return pd.NaT


def gest_age_to_list(gest_age: str) -> pd.Series:
    """
    Convert 'gest_age' string to list [stage, week/day, index]
    """
    try:
        gest_age = str(gest_age)
        if gest_age == "intro":
            return [0, 0, 0]
        data = gest_age.split(".")

        data[0] = int(data[0])
        data[1] = data[1][1:]

        if data[1][-1] == "M" or data[1][-1] == "m":
            data[1] = data[1][:-1]

        if len(data) == 3:
            if data[2][-1] == "M" or data[2][-1] == "m":
                data[2] = data[2][:-1]
        elif len(data) == 2:
            data.append(0)

        data[1] = int(data[1])
        data[2] = int(data[2])

        if data[0] in [5, 6]:
            data = [pd.NaT, pd.NaT, pd.NaT]

        data = pd.Series(
            {"stage": data[0], "week_day": data[1], "media_index": data[2]}
        )

        return data[0], data[1], data[2]
    except:
        if len(gest_age) > 300:
            return pd.NaT

        logging.warning(f"{gest_age} is not a valid gestation age.")
        return pd.NaT, pd.NaT, pd.NaT

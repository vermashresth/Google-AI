import logging
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import swifter
from pandas.api.types import CategoricalDtype

from .data_utils import calls_so_far, date_to_month, pad_sequences
from .utils import load_obj

ngo_hosp_dict = load_obj(os.path.join("prediction", "res", "ngo_hosp_dict.pkl"))
gest_dict = load_obj(os.path.join("prediction", "res", "gest_dict.pkl"))


def _days_to_first_call(row):
    if pd.isnull(row["startdate"]):
        return -1
    else:
        diff = row["startdate"] - row["registration_date"]
        if 0 <= diff <= 180:
            return diff
        else:
            return 8


def _calc_days_to_first_call(data, call_data):
    call_data = call_data[["user_id", "startdate", "callStatus"]]
    first_call_dates = (
        call_data[call_data["callStatus"] == 0].groupby("user_id")["startdate"].min()
    )
    data = pd.merge(data, first_call_dates, "left", left_on="user_id", right_index=True)
    data["days_to_first_call"] = data.parallel_apply(
        _days_to_first_call, axis=1
    )
    data = data.drop(columns=["startdate"])

    return data


def _preprocess_beneficiary_data(data, call_data):

    # calculate days to first connection
    data = _calc_days_to_first_call(data, call_data)

    # replace invalid values of each numeric column
    valid_ranges = {
        "enroll_gest_age": (1, 60, 14),
        "age": (15, 50, 25),
        "g": (1, 5, -1),
        "p": (0, 5, -1),
        "s": (0, 1, -1),
        "l": (0, 5, -1),
        "a": (0, 2, -1),
    }
    for col in valid_ranges.keys():
        min_val, max_val, replace_val = valid_ranges[col]
        data = data.round({col: 0})
        data.loc[~((min_val <= data[col]) & (data[col] <= max_val)), col] = replace_val

    # replace invalid values of each categorical column
    valid_categories = {
        "call_slots": ([1, 2, 3, 4, 5, 6], 3),
        "enroll_delivery_status": ([0, 1], 0),
        "language": ([2, 3, 4, 5], 2),
        "ChannelType": ([0, 1, 2], 0),
        "ngo_hosp_id": (ngo_hosp_dict.keys(), 4),
        "education": ([1, 2, 3, 4, 5, 6, 7], 3),
        "phone_owner": ([0, 1, 2], 0),
        "income_bracket": ([-1, 0, 1, 2, 3, 4, 5, 6], -1),
    }
    for col in valid_categories.keys():
        valid_vals, replace_val = valid_categories[col]
        data.loc[~data[col].isin(valid_vals), col] = replace_val
        data[col] = data[col].astype(CategoricalDtype(categories=valid_vals))

    # binning age into five groups
    bins = [14, 20, 25, 30, 35, 51]
    data.loc[:, "age"] = pd.cut(data["age"], bins=bins, labels=False)
    data["age"] = data["age"].astype(CategoricalDtype(categories=[0, 1, 2, 3, 4]))

    # use one hot encoding representation for categorical columns
    one_hot_columns = [
        "age",
        "language",
        "education",
        "phone_owner",
        "call_slots",
        "ChannelType",
        "income_bracket",
    ]
    data = pd.get_dummies(data, columns=one_hot_columns, prefix=one_hot_columns)

    # selecting columns and their order
    columns = [
        "user_id",
        "enroll_gest_age",
        "enroll_delivery_status",
        "ngo_hosp_id",
        "g",
        "p",
        "s",
        "l",
        "a",
        "days_to_first_call",
        "age_0",
        "age_1",
        "age_2",
        "age_3",
        "age_4",
        "language_2",
        "language_3",
        "language_4",
        "language_5",
        "education_1",
        "education_2",
        "education_3",
        "education_4",
        "education_5",
        "education_6",
        "education_7",
        "phone_owner_0",
        "phone_owner_1",
        "phone_owner_2",
        "call_slots_1",
        "call_slots_2",
        "call_slots_3",
        "call_slots_4",
        "call_slots_5",
        "call_slots_6",
        "ChannelType_0",
        "ChannelType_1",
        "ChannelType_2",
        "income_bracket_-1",
        "income_bracket_0",
        "income_bracket_1",
        "income_bracket_2",
        "income_bracket_3",
        "income_bracket_4",
        "income_bracket_5",
        "income_bracket_6",
    ]
    data = data[columns]

    data = data.dropna()

    return data


def _preprocess_call_data(data):

    # dropping rows will null values (ideally shouldn't drop any)
    data = data.dropna()

    # not checking duration as it is already checked during cleaning

    # checking callStatus
    data = data[data["callStatus"].isin([0, 1, 2])]

    # checking startdate (should be after 1 Jan 2018)
    data = data[data["startdate"] >= 0]

    # checking if all gest_ages are known
    for col in ["gest_stage", "gest_week_day", "gest_index"]:
        data[col] = data[col].astype("str")
    data["gest_age"] = data[["gest_stage", "gest_week_day", "gest_index"]].agg(
        ",".join, axis=1
    )

    # log all the unseen media IDs
    unseen_gest_ages = data[~data["gest_age"].isin(gest_dict.keys())]["gest_age"]
    for i in range(unseen_gest_ages.shape[0]):
        logging.warning("Unseen media ID: %s" % unseen_gest_ages.iloc[i])

    # use only seen gest_ages
    data = data[data["gest_age"].isin(gest_dict.keys())]
    data = data.drop(columns=["gest_age"])

    return data


def _build(beneficiaries, beneficiary_data, call_data, config):
    user_ids, static_xs, ngo_hosp_ids, dynamic_xs, gest_ages = [], [], [], [], []

    order = [
        "callStatus",
        "duration",
        "daysfrom",
        "month",
        "gest_stage",
        "gest_week_day",
        "gest_index",
    ]

    for i in range(beneficiaries.shape[0]):
        row = beneficiaries.iloc[i, :]

        # selecting the specific beneficiary's logs
        user_logs = call_data[call_data["user_id"] == row["user_id"]].copy()

        # calculating no of days from prediction day (start + input_length)
        user_logs.loc[:, "daysfrom"] = (
            row["start"] + config["input_length"] - user_logs["startdate"]
        )
        # calculating month to capture monthly patterns
        user_logs.loc[:, "month"] = user_logs["startdate"].apply(date_to_month)

        # selecting call logs before prediction day for input
        input_sequence = user_logs[
            user_logs["startdate"] <= row["start"] + config["input_length"]
        ]

        # checking if input satisfies minimum call requirements
        # if not, skipping the example
        if input_sequence.shape[0] < config["input_calls_thresh"]:
            continue

        # converting to numpy array
        input_sequence = np.array(input_sequence[order])

        # calculating features - no of attempts, connections, engagements
        attempts, connections, engagements = calls_so_far(
            row["user_id"], row["start"] + config["input_length"], call_data
        )

        # calculating features - days since last attempt, connection, engagement
        days_last_attempt = input_sequence[-1, 2]
        past_succ_calls = call_data[
            (call_data["user_id"] == row["user_id"])
            & (call_data["callStatus"] == 0)
            & (call_data["startdate"] <= row["start"] + config["input_length"])
        ]
        last_succ_conn = (
            past_succ_calls["startdate"].iloc[-1] if not past_succ_calls.empty else -1
        )
        last_succ_eng = (
            past_succ_calls[past_succ_calls["duration"] >= 30]["startdate"].iloc[-1]
            if not past_succ_calls[past_succ_calls["duration"] >= 30].empty
            else -1
        )
        days_last_connection = row["start"] + config["input_length"] - last_succ_conn
        days_last_engagement = row["start"] + config["input_length"] - last_succ_eng

        # selecting beneficiary features for the specific beneficiary
        beneficiary_features = np.array(
            beneficiary_data[beneficiary_data["user_id"] == row["user_id"]]
        )

        # saving user_id
        user_ids.append(beneficiary_features[0, 0])

        # concatenating all static inputs
        static_x = np.concatenate(
            [
                beneficiary_features[0, 1:3],
                beneficiary_features[0, 4:],
                np.array(
                    [
                        attempts,
                        connections,
                        engagements,
                        days_last_attempt,
                        days_last_connection,
                        days_last_engagement,
                    ]
                ),
            ],
            axis=0,
        )
        static_xs.append(static_x)

        # encoding the ngo_hosp_id for tensorflow embedding
        ngo_hosp_ids.append([ngo_hosp_dict[int(beneficiary_features[0, 3])]])

        # saving the call features as dynamic inputs
        dynamic_xs.append(input_sequence[:, :4])

        # saving the media id (gestation age) for each call
        gest_ages.append(
            np.array(
                [
                    gest_dict[",".join([str(int(x)) for x in gest_details])]
                    for gest_details in input_sequence[:, 4:]
                ]
            )
        )

    return (
        np.array(user_ids, dtype=np.int32),
        pad_sequences(dynamic_xs, n=config["input_calls_max"], dim=4),
        pad_sequences(gest_ages, n=config["input_calls_max"], dim=1),
        np.array(static_xs, dtype=np.int16),
        np.array(ngo_hosp_ids, dtype=np.int16),
    )


def _build_dataset(beneficiary_data, call_data):
    # configuration for generating features
    config = {
        "input_length": 60,
        "input_calls_thresh": 8,
        "input_calls_max": 18,
    }

    logging.info("%d beneficiaries in beneficiary data." % beneficiary_data.shape[0])

    # calculating start date, end date and attempt count for each beneficiary
    beneficiaries = (
        call_data.groupby("user_id")["startdate"]
        .agg([("start", "min"), ("end", "max")])
        .reset_index()
    )

    # calculating the number of days beneficiary is in the program
    beneficiaries.loc[:, "length"] = beneficiaries["end"] - beneficiaries["start"]
    # logging.info("beneficiaries: %s", beneficiaries)

    # filtering beneficiaries that do not meet minimum program length requirements
    beneficiaries = beneficiaries[beneficiaries["length"] >= config["input_length"]]

    logging.info(
        "%d beneficiaries have met %d-day program length requirement."
        % (beneficiaries.shape[0], config["input_length"])
    )
    logging.info("beneficiaries: %s", beneficiaries)

    # dividing the beneficiary and call data into chunks to parallelize dataset creation
    number_of_cores = 10
    beneficiaries_split = np.array_split(beneficiaries, number_of_cores)
    args = []
    for i in range(number_of_cores):
        beneficiaries_i = beneficiaries_split[i]
        call_data_i = call_data[call_data["user_id"].isin(beneficiaries_i["user_id"])]
        beneficiary_data_i = beneficiary_data[
            beneficiary_data["user_id"].isin(beneficiaries_i["user_id"])
        ]
        args.append((beneficiaries_i, beneficiary_data_i, call_data_i, config,))

    logging.info("gottem %s", args)
    # using multiprocessing to map processes to all available cores
    pool = Pool(number_of_cores)
    user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids = zip(
        *pool.starmap(_build, args)
    )
    pool.close()
    pool.join()

    # concatenating results of all processes
    user_ids = np.concatenate(user_ids, axis=0)
    dynamic_xs = np.concatenate(dynamic_xs, axis=0)
    gest_ages = np.concatenate(gest_ages, axis=0)
    static_xs = np.concatenate(static_xs, axis=0)
    ngo_hosp_ids = np.concatenate(ngo_hosp_ids, axis=0)

    # dynamic features preprocessing
    # normalizing duration
    dynamic_xs[:, :, 1] = dynamic_xs[:, :, 1] / 60
    dynamic_xs[:, :, 2] = dynamic_xs[:, :, 2] / 60
    dynamic_xs[:, :, 3] = dynamic_xs[:, :, 3] / 12

    # static features preprocessing
    # normalizing enroll gestation age
    static_xs[:, 0] = static_xs[:, 0] - 19.80987
    # normalizing days to first call
    static_xs[:, 7] = static_xs[:, 7] - 18.89268

    logging.info(
        "%d beneficiaries have met minimum %d calls requirement."
        % (beneficiary_data.shape[0], config["input_calls_thresh"])
    )

    dataset = (user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids)

    return dataset


def _select_relevant_data(beneficiary_data, call_data):
    # selecting beneficiaries that have call data
    beneficiary_data = beneficiary_data[
        beneficiary_data["user_id"].isin(call_data["user_id"])
    ]

    # selecting calls for selected beneficiaries
    call_data = call_data[call_data["user_id"].isin(beneficiary_data["user_id"])]

    return beneficiary_data, call_data


def preprocess_and_make_dataset(beneficiary_data, call_data):

    logging.info("Preprocessing beneficiary data.")

    beneficiary_data = _preprocess_beneficiary_data(beneficiary_data, call_data)

    logging.info("Preprocessing beneficiary data completed.")
    logging.info("Preprocessing call data.")

    call_data = _preprocess_call_data(call_data)

    logging.info("Preprocessing call data completed.")
    logging.info("Building the dataset.")

    # Selecting beneficiaries with available data
    beneficiary_data, call_data = _select_relevant_data(beneficiary_data, call_data)

    logging.info(
        "Preproccessed data has %d beneficiaries and %d calls."
        % (beneficiary_data.shape[0], call_data.shape[0])
    )

    dataset = _build_dataset(beneficiary_data, call_data)

    logging.info(
        "Dataset successfully build for %d beneficiaries." % (dataset[0].shape[0])
    )

    return dataset

import json
import logging

import numpy as np


def format_for_online_prediction(dataset):
    # unpacking the dataset
    user_ids, dynamic_xs, gest_ages, static_xs, ngo_hosp_ids = dataset

    # converting the examples to the required format
    instances = []
    instance_to_user_id = dict()
    for i in range(user_ids.shape[0]):
        instance = dict()
        instance["static"] = static_xs[i].tolist()
        instance["ngo_hosp_id"] = ngo_hosp_ids[i].tolist()
        instance["dynamic"] = dynamic_xs[i].tolist()
        instance["gest_age"] = gest_ages[i].tolist()
        instance_to_user_id[json.dumps(instance)] = user_ids[i]

        instances.append(instance)

    instances = {"instances": instances}

    return instances, instance_to_user_id

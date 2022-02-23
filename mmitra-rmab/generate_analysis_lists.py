import pandas as pd
from datetime import *

week_to_sdate_jan = {
    1: '2021-10-04',
2: '2021-10-11',
3: '2021-10-18',
4: '2021-10-25',
5: '2021-11-01',
6: '2021-11-08',
7: '2021-11-15',
8: '2021-11-22',
9: '2021-11-29',
10: '2021-12-06',
11: '2021-12-13',
12: '2021-12-20',
13: '2021-12-27',
14: '2022-01-03',
15: '2022-01-10',
16: '2022-01-17',
17: '2022-01-24',
18: '2022-01-31',
19: '2022-02-07',
20: '2022-02-14',
21: '2022-02-21'
}
week_to_sdate_april = {
    1: '2021-02-22',
2: '2021-03-01',
3: '2021-03-08',
4: '2021-03-15',
5: '2021-03-22',
6: '2021-03-29',
7: '2021-04-05',
8: '2021-04-12',
9: '2021-04-19',
10: '2021-04-26',
11: '2021-05-03',
12: '2021-05-10',
13: '2021-05-17',
14: '2021-05-24',
15: '2021-05-31',
16: '2021-06-07',
17: '2021-06-14',
18: '2021-06-21',
19: '2021-06-28',
20: '2021-07-05',
21: '2021-07-12',
22: '2021-07-19',
23: '2021-07-26',
24: '2021-08-02',
25: '2021-08-09',
26: '2021-08-16',
27: '2021-08-23',
28: '2021-08-30',
29: '2021-09-06',
30: '2021-09-13',
31: '2021-09-20',
32: '2021-09-27'
}
week_to_sdate_dicts = {"jan_data": week_to_sdate_jan, "feb16-mar15_data": week_to_sdate_april}
date_lists_jan = ['2021-10-04', '2021-10-11', '2021-10-18', '2021-10-25', '2021-11-01', '2021-11-08', '2021-11-15', '2021-11-22', '2021-11-29', '2021-12-06', '2021-12-13', '2021-12-20', '2021-12-27', '2022-01-03', '2022-01-10', '2022-01-17', '2022-01-24', '2022-01-31', '2022-02-07', '2022-02-14', '2022-02-21']
date_lists_april = ['2021-02-22', '2021-03-01', '2021-03-08', '2021-03-15', '2021-03-22', '2021-03-29', '2021-04-05', '2021-04-12', '2021-04-19', '2021-04-26', '2021-05-03', '2021-05-10', '2021-05-17', '2021-05-24', '2021-05-31', '2021-06-07', '2021-06-14', '2021-06-21', '2021-06-28', '2021-07-05', '2021-07-12', '2021-07-19', '2021-07-26', '2021-08-02', '2021-08-09', '2021-08-16', '2021-08-23', '2021-08-30', '2021-09-06', '2021-09-13', '2021-09-20', '2021-09-27']
date_lists = {"jan_data": date_lists_jan, "feb16-mar15_data":date_lists_april}
CONFIG = {
    'pilot_data': sys.argv[1],
    'current_week': int(sys.argv[2])
    }
CONFIG['pilot_dates'] = date_lists[CONFIG["pilot_data"]]
week_to_sdate = week_to_sdate_dicts[CONFIG["pilot_data"]]

def str_to_date(str_date):
    #return (pd.to_datetime(str_date, format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days
    try:
        y, m, d = [int(x) for x in str_date.split('-')]
        return date(y, m, d)
    except:
        return None
def date_to_week(xdate):
    for i in range(len(week_to_sdate),0,-1):
        if str_to_date(xdate) is None:
            return None
        if str_to_date(xdate) >= str_to_date(week_to_sdate[i]):
            return "week"+str(i)
# input files
df_exp = pd.read_csv(CONFIG["pilot_data"]+"/Experiment.csv")
df_int =pd.read_csv(CONFIG["pilot_data"]+"/interventions_data.csv")

#interventions file
df_int['intervene_week']=df_int.apply(lambda row: date_to_week(row['intervention_date']), axis=1)
print(df_int)
print(df_int[['beneficiary_id','intervention_date','intervene_week']])


df_int_processed = df_int
df_int_processed['user_id'] = df_int_processed['beneficiary_id'] # format fix
df_int_processed = df_int_processed[['user_id','intervention_date','intervene_week','intervention_id']]
print(df_int_processed)

print(df_exp)
df_joint = df_int_processed.set_index('user_id').join(df_exp.set_index('user_id'))
print(df_joint)

df_joint['user_id'] = df_joint.index 
df_joint['intervene_date'] = df_joint['intervention_date']


df_joint = df_joint[['user_id','intervene_week','intervene_date','intervention_id']]#,'days_since_reg']]
df_joint.to_csv(CONFIG["pilot_data"]+"/interventions.csv")

#analysis fle

import numpy as np
import pandas as pd
import ipdb
import pickle
from tqdm import tqdm
import sys
if CONFIG['pilot_data']=='jan_data':
    from training_new.utils import load_obj, save_obj
    from training_new.data import load_data
    from training_new.dataset import _preprocess_call_data, preprocess_and_make_dataset

    from training_new.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
else:
    from training.utils import load_obj, save_obj
    from training.data import load_data
    from training.dataset import _preprocess_call_data, preprocess_and_make_dataset

    from training.modelling.metrics import F1, Precision, Recall, BinaryAccuracy
    from tensorflow.keras.models import load_model

from tensorflow.keras.models import load_model


CONFIG["read_sql"] = 0
print(df_exp)

if CONFIG['pilot_data']=='jan_data':
    pilot_beneficiary_data, pilot_call_data = load_data(CONFIG)
else:
    pilot_beneficiary_data, pilot_call_data = load_data(CONFIG["pilot_data"])
pilot_call_data = _preprocess_call_data(pilot_call_data)


df_joint = df_exp.set_index('user_id').join(df_int_processed.set_index('user_id'))
df_joint['user_id'] = df_joint.index
df_joint['intervention_week'] = df_joint['intervene_week']
df_joint = df_joint[['user_id','registration_date','intervention_week']]
print(df_joint)
complete_group = df_joint
print("users with multiple interventions")
complete_group.index.name = None
dc = complete_group.groupby('user_id', as_index=False).count()
print(dc[dc['registration_date']>1])
complete_group = complete_group.sort_values('intervention_week')
complete_group = complete_group.drop_duplicates(subset='user_id', keep="last") # analysis wrt last intervention made
out_dict = {'user_id': [], 'registration_date': [], 'intervention_week': []}

for i in range(CONFIG['current_week']):
    out_dict['week{}_E/C'.format(i)] = []
    out_dict['week{}_state'.format(i)] = []
all_user_ids = complete_group['user_id'].to_list()

for user_id in tqdm(all_user_ids):
    out_dict['user_id'].append(user_id)
    curr_row = complete_group[complete_group['user_id'] == user_id]
    #print(curr_row)
    out_dict['registration_date'].append(curr_row['registration_date'].item())
    #out_dict['start_state'].append(curr_row['start_state'].item())
    out_dict['intervention_week'].append(curr_row['intervention_week'].item())

    curr_state = None
    countc, counte = 0, 0
    for i, date_val in enumerate(CONFIG['pilot_dates'][:CONFIG['current_week']]):
        pilot_date_num = (pd.to_datetime(date_val, format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days
        
        past_days_calls = pilot_call_data[
            (pilot_call_data["user_id"]==user_id)&
            (pilot_call_data["startdate"]<pilot_date_num)&
            (pilot_call_data["startdate"]>=pilot_date_num - 7)
        ]

        past_days_connections = past_days_calls[past_days_calls['duration']>0].shape[0]
        past_days_engagements = past_days_calls[past_days_calls['duration'] >= 30].shape[0]

        if past_days_engagements == 0:
            curr_state = 7
        else:
            curr_state = 6
    
        if out_dict['intervention_week'][-1]==out_dict['intervention_week'][-1]: #check for nan
            if i > int(out_dict['intervention_week'][-1][4:]):
                countc += past_days_connections
                counte += past_days_engagements
        
        out_dict['week{}_E/C'.format(i)].append('{}/{}'.format(past_days_engagements, past_days_connections))
    
        out_dict['week{}_state'.format(i)].append(curr_state)


df = pd.DataFrame(out_dict)
df.to_csv(CONFIG["pilot_data"]+'/all_analysis_week_{}.csv'.format(CONFIG['current_week']))

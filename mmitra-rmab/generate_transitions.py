import pandas as pd
from tqdm import tqdm
import sys
from datetime import *
week_to_sdate_jan = {
0: '2021-09-27',
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
0: '2021-02-15',
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




CONFIG={
    "analysis_file": sys.argv[1],
    "total_weeks":int(sys.argv[2]),
    "current_folder": sys.argv[3]
}
T = CONFIG["total_weeks"]
df = pd.read_csv(CONFIG["analysis_file"])
all_user_ids = df['user_id'].to_list()
week_to_sdate = week_to_sdate_dicts[CONFIG["current_folder"]]

def str_to_date(str_date):
    #return (pd.to_datetime(str_date, format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days
    try:
        y, m, d = [int(x) for x in str_date.split('-')]
        return date(y, m, d)
    except:
        return None

def date_to_week(xdate):
    for i in range(len(week_to_sdate)-1,0,-1):
        # print(xdate)
        if str_to_date(xdate) is None:
            print('Missing intervention date')
            return
        if str_to_date(xdate) >= str_to_date(week_to_sdate[i]):
            return i
    return None


out_dict = {'user_id': [], 'pre-action state': [], 'action': [], 'post-action state': [], 'start_date':[]}

intervention_dict = {}
df2 =  pd.read_csv(CONFIG["current_folder"]+'/interventions_data.csv')
count_all = 0
count = 0
max_week = 0
min_week = 99999
for user_id in all_user_ids:
    interventions_per_user = df2[df2['beneficiary_id']==user_id]
    dates_interventions_per_user = interventions_per_user['intervention_date'].to_list()
    if len(dates_interventions_per_user)==0:
        continue
    else:
        weeks_interventions_per_user = []
        for x in dates_interventions_per_user:
            week = date_to_week(x)
            if (week == week) and (week is not None): # check for nan/None
                weeks_interventions_per_user.append(week)

        intervention_dict[user_id] = weeks_interventions_per_user
        
        if len(weeks_interventions_per_user)>0:
            count+=1
            count_all += len(weeks_interventions_per_user)
            if min(weeks_interventions_per_user) < min_week:
                min_week = min(weeks_interventions_per_user)
            if max(weeks_interventions_per_user) > max_week:
                max_week = max(weeks_interventions_per_user)

print("Number of beneficiaries intervened: "+str(count))
print("Number of total interventions: "+str(count_all))
print("min_week: "+str(min_week))
print("max_week: "+str(max_week))

for user_id in tqdm(all_user_ids):
    curr_row = df[df['user_id'] == user_id]

    engagements = []
    for i in range(T):
        counte = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
        engagements.append(counte)

    if user_id in intervention_dict:
        user_intervention_list = intervention_dict[user_id]
        user_intervention_list.sort()
    else:
        user_intervention_list = []

    assert len(engagements) == T
    for i in range(T-1):
        start_state = 'L' if engagements[i] > 0 else 'H'
        next_state = 'L' if engagements[i + 1] > 0 else 'H'
        action = 'Intervention' if (i+1) in user_intervention_list else 'No Intervention'
        out_dict['user_id'].append(user_id)
        out_dict['pre-action state'].append(start_state)
        out_dict['action'].append(action)
        out_dict['post-action state'].append(next_state)
        out_dict['start_date'].append(week_to_sdate[i])

transitions_df = pd.DataFrame(out_dict)
print(transitions_df.groupby('action').count())
transitions_df.to_csv(CONFIG["current_folder"]+'/transitions.csv')

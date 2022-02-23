import pandas as pd
from tqdm import tqdm

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

CONFIG={
    "analysis_file": sys.argv[1],
    "total_weeks":sys.argv[2],
    "current_folder": sys.argv[3]
}
T = CONFIG["total_weeks"]
df = pd.read_csv(CONFIG["analysis_file"])
all_user_ids = df['user_id'].to_list()
week_to_sdate = week_to_sdate_dicts[CONFIG["current_folder"]]




out_dict = {'user_id': [], 'pre-action state': [], 'action': [], 'post-action state': [], 'start_date':[]}

intervention_dict = {}
for file in range(1,5):
    df2 =  pd.read_csv('outputs/interventions_week{}.csv'.format(file))
    users = df2['user_id'].to_list()
    for user_id in users:
            if user_id not in intervention_dict:
                intervention_dict[user_id] = [file+(T-5)]
            else:
                intervention_dict[user_id].append(file+(T-5))


for user_id in tqdm(all_user_ids):
    curr_row = df[df['user_id'] == user_id]

    engagements = []
    for i in range(T):
        counte = int(curr_row['week{}_E/C'.format(i)].item().split('/')[0])
        engagements.append(counte)

    if user_id in intervention_dict:
        user_intervention_list = intervention_dict[user_id]
    else:
        user_intervention_list = []

    assert len(engagements) == T
    for i in range(T-1):
        start_state = 'L' if engagements[i] > 0 else 'H'
        next_state = 'L' if engagements[i + 1] > 0 else 'H'
        action = 'Intervention' if i in user_intervention_list else 'No Intervention'
        out_dict['user_id'].append(user_id)
        out_dict['pre-action state'].append(start_state)
        out_dict['action'].append(action)
        out_dict['post-action state'].append(next_state)
        out_dict['start_date'].append(week_to_sdate[i])

transitions_df = pd.DataFrame(out_dict)
print(transitions_df.groupby('action').count())
transitions_df.to_csv(CONFIG["current_folder"]+'/transitions.csv')
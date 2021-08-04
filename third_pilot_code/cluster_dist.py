import pandas as pd
import matplotlib.pyplot as plt
import ipdb

CONFIG = {
    'calling_files': ['250_week1', '400_week2', '400_week3', '400_week4', '435_week5', '600_week6', '700_week7', '1000_week8']
}

rmab_group = pd.read_csv("outputs/pilot_outputs/rmab_pilot.csv")
rmab_user_ids = rmab_group['user_id'].values

clusters_per_week = { 'week1': [], 'week2': [], 'week3': [], 'week4': [], 'week5': [], 'week6': [] , 'week7': [], 'week8': []  }
intervention_dict = {}
for file in CONFIG['calling_files']:
    with open('intervention_lists/calling_list_{}.txt'.format(file), 'r') as fr:
        for line in fr:
            user_id = int(line.strip())
            if user_id in rmab_user_ids:
                if user_id not in intervention_dict:
                    intervention_dict[user_id] = [file.split('_')[1]]
                else:
                    intervention_dict[user_id].append(file.split('_')[1])
clusters = {}
for user_id in intervention_dict:
    cluster = rmab_group[rmab_group['user_id'] == user_id]['cluster'].item()
    weeks = intervention_dict[user_id]
    clusters[cluster] = clusters.get(cluster, 0) + len(weeks)
    for week in weeks:
        if cluster not in clusters_per_week[week]:
            clusters_per_week[week].append(cluster)

df = pd.DataFrame.from_dict(clusters, orient='index')
# df.sort_values('no. of times pulled', ascending=False)
df.to_csv("outputs/RMAB_clusters_pulled.csv")
plt.bar( list(clusters.keys()), list(clusters.values()) )
plt.xlabel("Cluster")
plt.ylabel("Number of times pulled until week 8")
plt.savefig("cluster_dist_week8.png")

num_cluster_per_week = {}
for week in clusters_per_week:
    num_cluster_per_week[week] = len(clusters_per_week[week])

clusters_until_week = {}
for i in range(8):
    clusters_until_week[i+1] = set(clusters_per_week[f'week{i+1}'])
    for j in range(i):
        clusters_until_week[i+1] = clusters_until_week[i+1].union(set(clusters_per_week[f'week{j+1}']))

num_cluster_until_week = {}
for week in clusters_until_week:
    num_cluster_until_week[week] = len(clusters_until_week[week])

ipdb.set_trace()
fig = plt.figure(figsize = (10, 5))

plt.bar( list(num_cluster_until_week.keys()), list(num_cluster_until_week.values()) )
plt.xlabel("Week")
plt.ylabel("Number of clusters pulled by RMAB until week")


plt.show()
# ipdb.set_trace()


import numpy as np
import matplotlib.pyplot as plt


predictive_loss = [
        [(624.4, 2.5), (433.8, 1.0), (453.2, 1.1)],
        [(1450.5, 0.7), (1285.5, 0.9), (1306.4, 1.0)],
        [(124.7, 0.5), (92.9, 0.2), (115.3, 1.0)],
        [(4384.2, 189.8), (1781.1, 4.0), (1888.7, 7.8)]
        ]

log_likelihood = [
        [(-624.4, 2.5), (-433.8, 1.0), (-453.2, 1.1)],
        [(-1450.5, 0.7), (-1285.5, 0.9), (-1306.4, 1.0)],
        [(-124.7, 0.5), (-92.9, 0.2), (-115.3, 1.0)],
        [(-4384.2, 189.8), (-1781.1, 4.0), (-1888.7, 7.8)]
        ]

eval_is = [
        [(320.0, 0.9), (334.7, 0.9), (345.9, 1.0)],
        [(360.7, 0.5), (362.4, 0.5), (363.1, 0.5)],
        [(301.7, 1.1), (313.0, 1.2), (323.9, 1.2)],
        [(1681.5, 4.2), (1679.9, 4.1), (1718.9, 4.3)]
        ]

eval_sim = [
        [(320.5, 0.9), (335.4, 0.9), (348.4, 0.9)],
        [(360.5, 0.5), (362.4, 0.4), (363.5, 0.5)],
        [(301.8, 1.1), (314.3, 1.2), (324.8, 1.2)],
        [(1716.6, 3.1), (1714.9, 3.0), (1733.8, 3.1)]
        ]

metric_list = [(log_likelihood, 'Log likelihood', 'likelihood', 0.02), (eval_is, 'Importance sampling evaluation', 'IS', 0.25), (eval_sim, 'Simulation-based evaluation', 'simulation', 0.2)]
# metric_list = [(predictive_loss, 'Predictive loss', 'loss', 0.02), (eval_is, 'Importance sampling evaluation', 'IS', 0.25), (eval_sim, 'Simulation-based evaluation', 'simulation', 0.2)]
directory = 'figs/barchart/'
domains = ['ARMMAN']
# domains = ['2-state', '5-state', '2-state partial', 'ARMMAN']

for metric, description, filename, margin in metric_list:
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
    for i, values in enumerate(metric[0:1]):
        ax = axs #[int(i/2), i%2]
        mean = [value[0] for value in values]
        ste  = [value[1] for value in values]
        ptp  = np.ptp(mean)
        ymin, ymax = min(mean) - ptp * margin, max(mean) + ptp * margin
        bottom = min((ymin, 0))
        print(bottom, mean[1], mean[2], ymin, ymax, ste[1], ste[2])

        # ax.bar([0, 1], [mean[1] - bottom, mean[2] - bottom], color=['#F4B400', '#4285F4'], yerr=[ste[1], ste[2]], bottom=bottom, width=0.5, edgecolor='black', linewidth=2, error_kw={'elinewidth': 3, 'capsize': 5})
        ax.bar([0,1,2], mean, color=['#DB4437', '#F4B400', '#4285F4'], yerr=ste, edgecolor='black', linewidth=2, error_kw={'elinewidth': 3, 'capsize': 5})
        # ax.set_title(description, fontsize=20)
        # ax.set_title(domains[i], fontsize=20)
        ax.set_xlim([-0.7, 2.7])
        ax.set_ylim([ymin, ymax])
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=16)

    fig.text(0.025, 0.5, description, fontsize=20, va='center', rotation='vertical')

    plt.savefig(directory + filename + '.png', bbox_inches='tight')
    plt.show()


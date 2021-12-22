class TrajectoryOps():
    def __init__(self, traj, policy_id, trial_id, T, n_benefs, cluster_ids, min_sup):
        self.traj = traj
        self.policy_id = policy_id
        self.trial_id = trial_id
        self.T = T
        self.n_benefs = n_benefs
        self.cluster_ids = cluster_ids
        self.min_sup = min_sup
        
        self.cluster_emp_prob_lookup = None
        self.benef_emp_prob_lookup = None
    
    def getBenefsEmpProbs(self, benef_ids, min_sup):
        benef_ci_traj = self.traj[self.trial_id, # trial index
                self.policy_id, # policy index
                :, # time index
                :, # tuple dimension
                benef_ids # benef index
            ]
        s_traj_c =  benef_ci_traj[:, :-1, dim_dict['state']]
        a_traj_c =  benef_ci_traj[:, :-1, dim_dict['action']]
        s_prime_traj_c =  benef_ci_traj[:, :-1, dim_dict['next_state']]
        a_prime_traj_c = benef_ci_traj[:, 1:, dim_dict['action']]

        transitions_df = pd.DataFrame(columns = ['s', 's_prime', 'a', 'a_prime'])

        for s_traj, a_traj, s_prime_traj, a_prime_traj in \
                            zip(s_traj_c, a_traj_c, s_prime_traj_c, a_prime_traj_c):
            transitions_df = transitions_df.append(pd.DataFrame({'s':s_traj,
                                    's_prime': s_prime_traj,
                                    'a': a_traj,
                                    'a_prime': a_prime_traj}), ignore_index=True)

        emp_prob, has_missing_values = getEmpProbs(transitions_df, min_sup)
                            
        return transitions_df, emp_prob, has_missing_values
    
    def getEmpProbs(self, transitions_df, min_sup = 1):
        emp_prob = {}
        has_missing_values = False
        for s in s_vals:
            for a in a_vals:
                s_a = transitions_df[(transitions_df['s']==s) &
                                        (transitions_df['a']==a)
                                    ]
                s_a_count = s_a.shape[0]
                key = (s, a)
                emp_prob[key] = {}
                for s_prime in s_vals:
                    for a_prime in a_vals:
                        s_a_s_prime_a_prime = s_a[(s_a['s_prime']==s_prime) &
                                                        (s_a['a_prime']==a_prime)
                                                    ]
                        s_a_s_prime_a_prime_count = s_a_s_prime_a_prime.shape[0]
                        options_key = (s_prime, a_prime)
                        if s_a_count >= min_sup:
                            emp_prob[key][options_key] = s_a_s_prime_a_prime_count/s_a_count
                        else:
                            emp_prob[key][options_key] = None
                            has_missing_values = True
                            
        return emp_prob, has_missing_values
    
    def getEmpProbClusterLookup(self):
        emp_prob_by_cluster = {}
        for cluster_id in np.unique(self.cluster_ids):
            benefs = getBenefsByCluster(cluster_id, self.cluster_ids)
            transitions_df, emp_prob, has_missing_values = self.getBenefsEmpProbs(benefs, self.min_sup)
            emp_prob_by_cluster[cluster_id] = emp_prob
        return emp_prob_by_cluster

    def getEmpProbBenefLookup(self):
        emp_prob_by_benef = {}
        for benef in tqdm.tqdm(range(self.n_benefs), desc='Emp Prob'):
            transitions_df, emp_prob, has_missing_values = self.getBenefsEmpProbs([benef], self.min_sup)
            emp_prob_by_benef[benef] = emp_prob
        return emp_prob_by_benef
    
    def imputeEmpProb(self, options):
        for k in options:
            if options[k]==None:
                options[k] = 0.25
        return options 

    def computeEmpProbs(self, cluster_level=True):
        if cluster_level:
            self.cluster_emp_prob_lookup = self.getEmpProbClusterLookup()
        else:
            self.benef_emp_prob_lookup = self.getEmpProbBenefLookup()
    
    def augmentTraj(self, lookup_by_cluster, n_aug_traj):
        aug_traj = np.zeros((n_aug_traj, 1, self.T-1, len(dim_dict), self.n_benefs))

        for aug_traj_i in tqdm.tqdm(range(n_aug_traj), desc='Augment Trajectory'):
            for benef in range(self.n_benefs):
                s, a = self.traj[self.trial_id, # trial index
                    self.policy_id, # policy index
                    0, # time index
                    [dim_dict['state'], dim_dict['action']], # tuple dimension
                    benef # benef index
                ]
                if lookup_by_cluster:
                    if not self.cluster_emp_prob_lookup:
                        raise 'Cluster Level Emp probs not yet computed. Run cluster_emp_prob_lookup method first.'
                    benef_cluster = self.cluster_ids[benef]
                    emp_prob = self.cluster_emp_prob_lookup[benef_cluster]
                else:
                    if not self.benef_emp_prob_lookup:
                        raise 'Cluster Level Emp probs not yet computed. Run cluster_emp_prob_lookup method first.'
                    
                    emp_prob = self.benef_emp_prob_lookup[benef]

                for ts in range(self.T-1):
                    options = emp_prob[(s, a)]
                    options = imputeEmpProb(options)
                    choice = np.random.choice(np.arange(len(list(options.keys()))),
                                                p=list(options.values()))
                    s_prime, a_prime = list(options.keys())[choice]
                    aug_traj[aug_traj_i, 0, ts, dim_dict['state'], benef] = s
                    aug_traj[aug_traj_i, 0, ts, dim_dict['action'], benef] = a
                    aug_traj[aug_traj_i, 0, ts, dim_dict['next_state'], benef] = s_prime
                    aug_traj[aug_traj_i, 0, ts, dim_dict['reward'], benef] = s
                    s, a = s_prime, a_prime
        print('Generated Augmented Traj of shape: ', aug_traj.shape)
        return aug_traj

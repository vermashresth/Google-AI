
# policy_names={0:'random',1:'rr', 2:'whittle'}
policy_names={0:'random',1:'rr', 2:'whittle', 3:'soft-whittle'}
policy_map = {policy_names[key]:key for key in policy_names}
dim_dict = {'state':0, 'action':1, 'next_state':2, 'reward':3}

S_VALS = [0, 1]
A_VALS = [0, 1]

N_STATES = len(S_VALS)
N_ACTIONS = len(A_VALS)

E_START_STATE_PROB_ARMMAN = 0.5

ARMMAN_FEAT_NAMES = [
        "enroll_gest_age",
        "enroll_delivery_status",
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


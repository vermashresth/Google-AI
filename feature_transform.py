from sklearn import preprocessing
import pandas as pd

def prepare_feat_df(pilot_static_features, pilot_user_ids):
    feat_len = pilot_static_features.shape[1]
    feat_cols = [f'f{i}' for i in range(feat_len)]
    pilot_feat_df = pd.DataFrame(columns= ['user_id'] + feat_cols)
    pilot_feat_df['user_id'] = pilot_user_ids

    scaler = preprocessing.StandardScaler().fit(pilot_static_features)
    pilot_feat_df[feat_cols] = scaler.transform(pilot_static_features)
    pilot_feat_df = pilot_feat_df.set_index('user_id').loc[pilot_user_ids]

    return pilot_feat_df, feat_cols
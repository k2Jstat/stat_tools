def predict_lgbregCV(X,y,rs = 71,inf = -99999999,sup = 99999999):
    """
    input   : pandas DataFrame
    output  : predict_value
              model_list
              
    example : predict_lgbreg(feature,label)
    """

    import random
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    def make_fold(var):
        if (0 <= var) & (var <= 0.33):
            return 1
        elif (0.33 < var) & (var <= 0.66):
            return 2
        else:
            return 3
    
    X_,y_ = X.copy(),y.copy()
    
    np.random.seed(rs)
    X_["random"] = pd.Series( np.random.random( len(X_) ,), index=X_.index )

    X_["fold"] = X_["random"].apply(make_fold)

    reg_list = []
    oof_preds = np.zeros(len(X_))
    importances = pd.DataFrame()

    for n_fold in range(3):
        #print(n_fold + 1)
        trn_ = X_[X_["fold"] != n_fold+1].index
        val_ = X_[X_["fold"] == n_fold+1].index

        X_tra,y_tra = X_.iloc[trn_],y_.iloc[trn_]
        X_val,y_val = X_.iloc[val_],y_.iloc[val_]

        reg = lgb.LGBMRegressor(max_depth = 3,learning_rate=0.05,rondom_state = rs)

        reg.fit(X_tra,y_tra,
                eval_metric="RMSE",
                eval_set= [(X_tra, y_tra), (X_val, y_val)], 
                early_stopping_rounds = 10,
                verbose = 10,
           )

        reg_list.append(["reg_" + str(n_fold + 1),reg])

        oof_preds[val_] = reg.predict(X_val, num_iteration=reg.best_iteration_)

        imp_df = pd.DataFrame()
        imp_df['feature'] = X_tra.columns
        imp_df['gain'] = reg.feature_importances_
        imp_df['fold'] = n_fold + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

    plt.figure(figsize=(8, 4))
    sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    #plt.savefig('importances.png')

    temp = pd.DataFrame()
    temp["pred_temp"] = oof_preds
    temp["inf"] = inf
    temp["sup"] = sup
    
    temp["pred_value"] = temp[["pred_temp","inf"]].max(axis = 1)
    temp["pred_value"] = temp[["pred_value","sup"]].min(axis = 1)
    
    temp = temp.drop(["inf","sup","pred_temp"],axis = 1)
    oof_preds2 = temp["pred_value"]
    
    print('RMSE : %.5f ' % np.sqrt(mean_squared_error(y,oof_preds)))
    return oof_preds2 , reg_list
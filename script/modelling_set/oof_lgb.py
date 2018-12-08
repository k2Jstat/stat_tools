def oof_lgb(X,y, bt = "gbdt", itr = 500, k_fold = 3,rs = 71, lr= 0.1,vb = 100,es = 10,param = None):
    """
    input   : feature,label
    output  : oof_preds,clf_list

    example : oof_lgb(train,"target") 
    """
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold


    #ターゲット変数と、除外特徴量はモデルに取り込まない
    #probabilityを付与した評価データを保存する用のデータフレーム
    clf_list = []
    pred_sum = 0
    oof_preds = np.zeros((len(X), len(sorted(y.unique()))))
    
    #closs_validation
    for n_fold, (trn_idx, val_idx) in enumerate(StratifiedKFold(n_splits=k_fold,shuffle=True,random_state=rs).split(X,y)):
        trn_x, trn_y = X.iloc[trn_idx].copy(), y.iloc[trn_idx].copy()
        val_x, val_y = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

        clf = lgb.LGBMClassifier(
            boosting_type = bt,
            random_state  = rs,
            n_estimators  = itr,
            learning_rate = lr,
            n_jobs = 4
        )
        #設定パラメータがあれば、パラメータを利用する
        if param != None:
            clf.set_params(**param)
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric=lgb_multi_weighted_logloss,      
                verbose=vb, 
                early_stopping_rounds=es
               )
        
        #validデータに予測値を付与
        oof_preds[val_idx, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('fold',n_fold+1,multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_) ) )

        #モデルの保存
        clf_list.append(["clf_" + str(n_fold + 1),clf])
        
        #oofで付与した予測値データセットを作成
        #out_df = pd.concat([out_df,val_x],axis = 0)
        gc.collect()
        
    #oofで付与した予測値の精度
    loss = multi_weighted_logloss(y_true=y, y_preds=oof_preds)
    print('MULTI WEIGHTED LOG LOSS : %.5f ' % loss)
    
    #return out_df, clf_list
    return oof_preds,clf_list

def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss
    
def predict_average(X,clf_list,pred_name = "class_"):
    """
    input   : pandas Dataframe
    output  : pandas Dataframe predict_value

    example : predict_average(feature,clf_list)
    """
    for i in range(len(clf_list)):
        if i == 0:
            temp = clf_list[i][1].predict_proba(X)
        else :
            temp += clf_list[i][1].predict_proba(X)
    temp = temp/len(clf_list)
    temp = pd.DataFrame(temp)
    temp.columns = [pred_name + str(i + 1) for i in temp.columns]
    return temp


def predict_lassoCV(X,y,reg_type = "lasso",inf = -999999,sup = 999999,cv = 10):
    """
    input   : pandas DataFrame
    output  : predict_value
              model
              
    example : predict_lassoCV(feature,label)
    """

    from sklearn.linear_model import LassoCV
    import pandas as pd
    #目的変数に対して、回帰式を設定
    reg = LassoCV(cv = cv,random_state = 12345,normalize=True,positive = True)
    #reg = RidgeCV(cv = cv,normalize=True)
    #reg = ElasticNetCV(cv = cv)
    
    reg.fit(X,y)
    
    X2 = pd.DataFrame()
    X2["pred_temp"] = reg.predict(X)
    X2["inf"] = inf
    X2["sup"] = sup
    
    X2["pred_value"] = X2[["pred_temp","inf"]].max(axis = 1)
    X2["pred_value"] = X2[["pred_value","sup"]].min(axis = 1)
    
    X2 = X2.drop(["inf","sup","pred_temp"],axis = 1)
    print("RMSE : " + str(np.sqrt(mean_squared_error(y,X2))))
    
    return X2["pred_value"],reg
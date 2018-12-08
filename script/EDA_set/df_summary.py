def df_summary(df):
    """
    input   : pandas DataFrame
    output  : count,unique,top,freq,mean,std,
              percentiles = {min,5%,25%,50%,75%,95%,max}, 
              missing_count,all,unique_count,dtypes

    example : df_summary(train)
    """

    import pandas as pd

    print("obs : " + str(df.shape[0]))
    print("columns : " + str(df.shape[1]))
    
    # pandas describe
    temp1 = df.describe(include = "all",percentiles={.05,.25,.50,.75,.95}).T
    # missing count
    temp2 = pd.DataFrame(len(df) - df.count())
    temp2.columns = ["missing_count"]
    temp3 = temp1.join(temp2)
    # all count
    temp3["all"] = temp3["count"] + temp3["missing_count"] 
    # unique count
    temp4 = pd.DataFrame(df.nunique(dropna = False))
    temp4.columns = ["unique_count"]
    temp5 = temp3.join(temp4)
    # dtypes
    temp6 = pd.DataFrame(df.dtypes)
    temp6.columns = ["dtypes"]    
    temp7 = temp5.join(temp6)
    
    return temp7

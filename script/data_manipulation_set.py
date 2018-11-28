def check_duplicates(df,key):
    """
    input   : pandas DataFrame
    output  : DataFrame duplicated by key

    example : check_duplicates(train,"id")
    """
    all_idx = df.index
    unique_idx = df.drop_duplicates(key,keep = "first").index
    dup_idx = list(set(all_idx) - set(unique_idx)) 
    
    duplicate_df = df.iloc[dup_idx]

    return duplicate_df
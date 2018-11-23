def dist_by_cat(df,category,yoko = 6,tate = 6,yoko_block = 5,plot_type = "violin"):
    """
    input   : pandas DataFrame
    output  : violin_plot 
              box_plot
              dist_plot

    example : dist_by_cat(train,"target")  
    """

    import matplotlib.pyplot as plt
    import math
    import seaborn as sns
    import numpy as np
    import pandas as pd

    #set the numerical features
    num_feat = [f for f in df.columns if df[f].dtype != 'object']
    
    print("★ " + category)
    # violin plot
    if plot_type == "violin":
        plt.figure(figsize = (yoko * yoko_block, tate * math.ceil(len(num_feat) / yoko_block)))
        for i in range(len(num_feat)):   
            plt.subplot(math.ceil((len(num_feat))/ yoko_block) + 1, yoko_block, i+1)
            sns.violinplot(data = df,y = num_feat[i],x = category,inner="quartile",split = True)
        plt.show()

    # box plot
    elif plot_type == "box":
        plt.figure(figsize = (yoko * yoko_block, tate * math.ceil(len(num_feat) / yoko_block)))
        for i in range(len(num_feat)):   
            plt.subplot(math.ceil((len(num_feat))/ yoko_block) + 1, yoko_block, i+1)
            sns.boxplot(data = df,y = num_feat[i],x = category)
        plt.show()

    # dist plot
    elif plot_type == "dist":
        plt.figure(figsize = (yoko * yoko_block, tate * math.ceil(len(num_feat) / yoko_block)))
        for i in range(len(num_feat)):
            plt.subplot(math.ceil(len(num_feat) / yoko_block), yoko_block, i+1)
            category_list = list(np.sort(df[category].unique()))
            for j in category_list:
                sns.distplot(df[df[category] == j].iloc[:,i].dropna().copy())
                #plt.legend()
            #fig = plt.subplots(int(len(train.columns)/2),2)
        plt.show()

    else :
        print("★ choose the following plot_type")
        print("★ 'violin' or 'box' or 'dist' ")

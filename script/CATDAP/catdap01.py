class CATDAP01:

    def __init__(self,X_bins = 5,y_type = "class",y_bins = 5,thresholds_category_count = 50,adj = 1e-10):
        self.X_bins = X_bins
        self.y_type = y_type
        self.y_bins = y_bins
        self.thresholds_category_count = thresholds_category_count
        self.adj = adj

    def fit(self,X,y):
        """
        input/X,y
        example/CATDAP().fit(X=X_train,y = train_label)
        """
        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        #outputobjectの作成
        AIC_LIST = []
        CROSS_DICTIONARY = {}

        feature_list = X.columns

        # AICを計算　*******************
        for feature in tqdm(feature_list):
            #print(feature)
            #対象データの作成
            #目的変数がカテゴリの場合はそのまま使用
            if self.y_type == "class":
                DF = pd.DataFrame(data = {"feature":X[feature],"label":y})

            #目的変数が数値の場合はの場合はそのまま使用
            elif self.y_type == "numeric":
                if self.y_bins == None:
                    print("If y_type is class ,set integer into y_bins")
                    return

                else :
                    #数値の分割
                    y_num = pd.qcut(y,self.y_bins,duplicates = "drop")
                    #数値の大小でソートして、ラベルを大小順にわかるようにふよ
                    labels_cut = list(y_num.unique().sort_values().astype(str))

                    for i in range(0,len(labels_cut)):
                        #とりあえず99分割までは対応させる。
                        i0 = str((101 + i))[1:3]
                        y_num = y_num.astype(str)
                        y_num.replace({labels_cut[i] : i0 + "_" +  labels_cut[i]},inplace = True)

                    DF = pd.DataFrame(data = {"feature":X[feature],"label":y_num})


            DF["label"] = DF["label"].astype(str)

            #欠損データ、非欠損データに分割
            DF_miss = DF[pd.isna(DF["feature"]) == True].copy()
            DF_nmiss = DF[pd.isna(DF["feature"]) == False].copy()

            #変数が時間の場合、数値に修正してやる
            if str(DF_nmiss["feature"].dtype).find("[ns]") > -1:
                DF_nmiss["feature"] = DF_nmiss["feature"].astype(int)

            #カテゴリ変数はそのままカテゴリとして採用
            if DF_nmiss["feature"].dtype == "object":
                if len(DF_nmiss["feature"].unique()) >= self.thresholds_category_count:
                    DF_nmiss["category"] = "All"

                else :
                    DF_nmiss["category"] = DF_nmiss["feature"]

            #数値データはビン化
            #if (DF_nmiss["feature"].dtype == "float") | (DF_nmiss["feature"].dtype == "int") :
            else :
                try :
                    DF_nmiss["category"] = pd.qcut(DF_nmiss["feature"],self.X_bins,duplicates = "drop")

                #基本50分割でセット。それで分割できなければもう生の値でいい
                except :
                    if len(DF_nmiss["feature"].unique()) >= self.thresholds_category_count:
                        DF_nmiss["category"] = "All"
                    else :
                        DF_nmiss["category"] = DF_nmiss["feature"]

            #else :
            #    print("予想外の事態")

                cat_list = list(DF_nmiss["category"].unique())
                cat_list.sort()

                cat_list2 = {}
                for i in range(len(cat_list)):
                        #print(str(i) + ": " + str(cat_list[i]))
                        i0 = str(100 + i + 1)
                        cat_list2[cat_list[i]] = i0[1:3] + ": " + str(cat_list[i])

                DF_nmiss["category"].replace(cat_list2,inplace = True)

            #else :
            #    print("予想外の事態")

            #欠損先はカテゴリを欠損で埋める
            DF_miss["category"] = "00: Deficiency"

            #欠損先と入力先を縦積み
            DF2 = pd.concat([DF_nmiss,DF_miss],axis = 0)

            #AICを計算するためのクロス表を作成
            _cross = pd.crosstab(DF2["category"],DF2["label"],margins=None,dropna = False)
            #該当先がいない場合、NAになるので0埋め
            _cross = _cross.fillna(0)

            n_feature , n_label = _cross.shape
            #全体件数での対数尤度。log内が0だと計算できないので0.00000000001を足してやる

            _term2 = (_cross.sum().sum()) * np.log(_cross.sum().sum() + self.adj)

            #AIC0 =====================================================================================
            temp = []
            #列単位での和
            for i in range(0,n_feature):
                temp.append((_cross.iloc[i,:].sum()) * np.log(_cross.iloc[i,:].sum() + self.adj) )

            #行単位での和
            for j in range(0,n_label):
                temp.append((_cross.iloc[:,j].sum()) * np.log(_cross.iloc[:,j].sum() + self.adj))

            _term1_0 = sum(temp)

            AIC0 = -2 * (_term1_0 - 2 * _term2) + 2 * (n_feature + n_label - 2)

            #AIC1 =====================================================================================
            _cross_yudo = np.zeros([n_feature ,n_label])

            for i in range(0,n_feature):
                for j in range(0,n_label):
                    _cross_yudo[i,j] = (_cross.iloc[i,j]) * np.log(_cross.iloc[i,j] + self.adj)

            _term1_1 = _cross_yudo.sum()

            AIC1 = -2 * (_term1_1 - _term2) + 2 * (n_feature * n_label - 1)

            #CATのAIC ==================================================================================
            AIC_CAT = AIC1 - AIC0

            #情報の入力率
            input_rate = len(DF_nmiss)/(len(DF_nmiss) + len(DF_miss))

            #クロス表
            _cross2 = pd.crosstab(DF2["category"],DF2["label"],margins=True,dropna = False)

            #各クラスの構成比を計算
            loop = len(_cross2.columns) - 1
            for i in range(0,loop):
                _cross2["Ratio_" + str(_cross2.columns[i])] = _cross2.iloc[:,i]/_cross2["All"]

            #計算したAICをリストに保存
            _OUT = [feature,AIC_CAT,input_rate]
            AIC_LIST.append(_OUT)
            CROSS_DICTIONARY[feature] = _cross2
        # ここまでAICの計算 ****************

        #資料を保存
        AIC_LIST2 = pd.DataFrame(AIC_LIST)
        AIC_LIST2.columns = ["Feature","AIC","Input_Ratio"]
        AIC_LIST3 = AIC_LIST2.sort_values(["AIC"])

        self.AIC_LIST = AIC_LIST3
        self.CrossTable_dict = CROSS_DICTIONARY

    def CrossTable_toExcel(self,file_name = "CrossTab_AIC.xls",path_output = "./"):
        """
        input/file_name,path_output
        example/CATDAP().CrossTable_toExcel(file_name = "temp.xls",path_output = "../output/")
        """

        base = pd.DataFrame()

        #for key in self.CrossTable_dict.keys():
        for key in self.AIC_LIST["Feature"]:
            temp = pd.DataFrame(data = self.AIC_LIST[self.AIC_LIST["Feature"] == key]["AIC"])
            temp.index.name = ["category"]
            temp.index = [key]
            temp = pd.concat([temp,self.CrossTable_dict[key]],sort = False)
            base = pd.concat([base,temp])

        base.fillna("",inplace = True)
        #吐き出し
        base.to_excel(path_output + file_name)
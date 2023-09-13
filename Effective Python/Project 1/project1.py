import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate # To display information
from joblib import dump # To save models
import os # To save models
# Imports libraries used

def ask_if_regressor_or_classifier(): #Gather data and ask if regressor or classifier function
    ask = int(input("Enter \n1 if you want regressor \n2 if you want classifier\n: ")) # Asks user if they want regressor or classifier
    if ask == 1: # If the user chose regressor
        regressor_csv = input("You chose regressor. Now please enter path for the .csv file\n: ") # Asks user to enter the path for the csv file
        df = pd.read_csv(regressor_csv) # Reads the input and creates df
        pd.set_option('display.max_columns', None) # Make sures all columns are displayed in case there's many columns
        print("\nPrinting dataframe head\n",df.head(),"\n") #Prints the head of df
        dependent_target = input("Please enter the column name for your dependent target (y)\n: ") # Asks user to enter the name of the column that will be their dependent target
        y = df.loc[:,dependent_target] # Creates y based on dependent target
        X = df.drop(dependent_target,axis=1) # Creates X by dropping dependent target from df

        df_null_data = df.isnull().sum().sum() # Checks for missing data in df
        if df_null_data >= 1:
            print(f"""There is missing data in the dataframe, fill in the data and rerun the app
These are the columns/column with missing data:\n{df.isnull().sum()}""") # Prints the columns with missing data
            exit()

        if X.select_dtypes(exclude=["number"]).empty == True and y.dtype == "float64" or y.dtype == "int64" or y.dtype == "uint64": # Checks if data is continuous
            print("Data is good for machine learning stage")
            return ("regressor",y,df,dependent_target,X) # If continuous returns these variables

        else:
            print("There is categorical data in your dataframe. Maybe try classifier instead. \nExiting") # Exits the app if categorical data is found
            exit()
            
    elif ask == 2: # If the user chose classifier
        from sklearn.utils.multiclass import type_of_target # Helps later for checking if the data is okay for classifier models
        classifier_csv = input("You chose classifier. Now please enter path for the .csv file\n: ") # Same as before
        df = pd.read_csv(classifier_csv)
        pd.set_option('display.max_columns', None)
        print("\nPrinting dataframe head\n",df.head(),"\n")
        dependent_target = input("Please enter the column name for your dependent target (y)\n: ")
        y = df.loc[:,dependent_target]
        X = df.drop(dependent_target,axis=1)
        y_type = type_of_target(y,input_name="y") # Creates a y_type based on y

        df_null_data = df.isnull().sum().sum() # Same as before
        if df_null_data >= 1:
            print(f"There is missing data in the dataframe, fill in the data and rerun the app. \nThese are the columns/column with missing data:\n{df.isnull().sum()}")
            exit()

        if X.select_dtypes(exclude=["number"]).empty == True and y.dtype == "object": # If X is continuous but y is categorical
            print("Data is good for machine learning stage")
            return ("classifier",y,df,dependent_target,X) # Returns these variables

        elif y_type in "binary" and X.select_dtypes(exclude=["number"]).empty == False: # If y is binary and X contains categorical data
            dummies = int(input("""You have categorical data in your dataframe, you need to create dummies to move on to machine learning stage. 
Would you like to create dummies? 
1-Yes 
2-No
: """)) # Asks user if they want to create dummies for the categorical data
            
            if dummies == 1: # Creates dummies if they answer yes
                print("*** Creating dummies ***")
                X = pd.get_dummies(df.drop(dependent_target,axis=1),drop_first=True)
                time.sleep(2)
                print("*** Dummies have now been created ***")
                print("Data is good for machine learning stage")
                return ("classifier",y,df,dependent_target,X) # Then returns these variables because data is okay to move on

            if dummies == 2:
                print("Exiting app")
                exit()

        elif y_type not in "binary" and X.select_dtypes(exclude=["number"]).empty == False: # If y not binary (like different species) and X contains categorical data
            dummies = int(input("""You have categorical data in your dataframe, you need to create dummies to move on to machine learning stage. 
Would you like to create dummies?
1-Yes
2-No
: """)) # Same as previous one
            
            if dummies == 1:
                print("*** Creating dummies ***")
                X = pd.get_dummies(df.drop(dependent_target,axis=1),drop_first=True)
                time.sleep(2)
                print("*** Dummies have now been created ***")
                print("Data is good for machine learning stage")
                return ("classifier",y,df,dependent_target,X)

            if dummies == 2:
                print("Exiting app")
                exit()

        elif y_type in "binary": # If y is binary and X does not contain categorical data
            print("Data is good for machine learning stage")
            return ("classifier",y,df,dependent_target,X) # Returns these variables

        elif X.select_dtypes(exclude=["number"]).empty == True and y_type not in "binary": # X is continuous and y is not in binary
            print("Your data is continuous, and your dependent target is not binary. This will not work with classifier models.\nExiting") 
            # Tells the user it wont work with classifier models and exits
            exit()

    else:
        exit()

def machine_learning(): #The machine learning function
    if regressor_or_classifier == "regressor": # If user wanted regressor models they come here
        from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
        from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
        from sklearn.svm import SVR
        # Imports relevant models and scoring for the regressor models

        mydata = [[],[],[],[],[],[]] #List for my table to display information
        rowindex = ["LiR","Lasso","Ridge","Elastic Net","SVR","Best Score"] #Rows for my table to display information
        head = ["","Best Parameters","MAE","RMSE","R2 score"] #Headers for my table to display information

        print("*** Beginning machine learning stage for regressor ***")
        print("\nThe models we'll use to test your data are:\nLiR\nLasso\nRidge\nElastic Net\nSVR\n* This may take some time so please have patience *")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101) # Trains and splits data
        
        # Linear Regression
        LiR_model = LinearRegression() # Creates the model
        LiR_model.fit(X_train,y_train) # Fits the model
        LiR_prediction = LiR_model.predict(X_test) # Prediction
        LiR_parameters = {"n_jobs":[1,2,5,8,10,25,50,100,250,1000]} # The parameters for Gridsearch

        LiR_MAE = mean_absolute_error(y_true=y_test,y_pred=LiR_prediction) # Calculates MAE
        LiR_RMSE = np.sqrt(mean_squared_error(y_true=y_test,y_pred=LiR_prediction)) # Calculates RMSE
        LiR_CV10_grid = GridSearchCV(estimator=LiR_model,param_grid = LiR_parameters,scoring="neg_mean_squared_error",cv=10) #Gridsearch finds best parameters
        LiR_CV10_grid = LiR_CV10_grid.fit(X_train,y_train) # Fits the Gridsearch model
        LiR_R2_score = r2_score(y_true=y_test,y_pred=LiR_prediction) # Calculates R2 score

        # Adds the models scores to my table I'll use to display the information with
        mydata[0].append(LiR_CV10_grid.best_params_)
        mydata[0].append(LiR_MAE)
        mydata[0].append(LiR_RMSE)
        mydata[0].append(LiR_R2_score)

        # Then everything repeats for the rest of the models
        # Lasso
        Lasso_model = Lasso()
        Lasso_model = Lasso_model.fit(X_train,y_train)
        Lasso_prediction = Lasso_model.predict(X_test)
        Lasso_parameters = {"alpha":[0.1,0.2,0.5,1,10,50,100,200]}
        
        Lasso_MAE = mean_absolute_error(y_true=y_test,y_pred=Lasso_prediction)
        Lasso_RMSE = np.sqrt(mean_squared_error(y_true=y_test,y_pred=Lasso_prediction))
        Lasso_CV10_grid = GridSearchCV(estimator=Lasso_model,param_grid=Lasso_parameters,scoring="neg_mean_squared_error",cv=10)
        Lasso_CV10_grid = Lasso_CV10_grid.fit(X_train,y_train)
        Lasso_R2_score = r2_score(y_true=y_test,y_pred=Lasso_prediction)

        mydata[1].append(Lasso_CV10_grid.best_params_)
        mydata[1].append(Lasso_MAE)
        mydata[1].append(Lasso_RMSE)
        mydata[1].append(Lasso_R2_score)

        # Ridge
        Ridge_model = Ridge()
        Ridge_model = Ridge_model.fit(X_train,y_train)
        Ridge_prediction = Ridge_model.predict(X_test)
        Ridge_parameters = {"alpha":[0.1,0.2,0.5,1,10,50,100,200]}
        
        Ridge_MAE = mean_absolute_error(y_true=y_test,y_pred=Ridge_prediction)
        Ridge_RMSE = np.sqrt(mean_squared_error(y_true=y_test,y_pred=Ridge_prediction))
        Ridge_CV10_grid = GridSearchCV(estimator=Ridge_model,param_grid=Ridge_parameters,scoring="neg_mean_squared_error",cv=10)
        Ridge_CV10_grid = Ridge_CV10_grid.fit(X_train,y_train)
        Ridge_R2_score = r2_score(y_true=y_test,y_pred=Ridge_prediction)

        mydata[2].append(Ridge_CV10_grid.best_params_)
        mydata[2].append(Ridge_MAE)
        mydata[2].append(Ridge_RMSE)
        mydata[2].append(Ridge_R2_score)
        
        # Elastic Net
        Elastic_Net_model = ElasticNet()
        Elastic_Net_model = Elastic_Net_model.fit(X_train,y_train)
        Elastic_Net_prediction = Elastic_Net_model.predict(X_test)
        Elastic_Net_parameters = {"alpha":[0.1,0.2,0.5,1,10,50,100,200],"l1_ratio":[0.1,0.5,0.7,0.9,0.95,0.99,1]}
        
        Elastic_Net_MAE = mean_absolute_error(y_true=y_test,y_pred=Elastic_Net_prediction)
        Elastic_Net_RMSE = np.sqrt(mean_squared_error(y_true=y_test,y_pred=Elastic_Net_prediction))
        Elastic_Net_CV10_grid = GridSearchCV(estimator=Elastic_Net_model,param_grid=Elastic_Net_parameters,scoring="neg_mean_squared_error",cv=10)
        Elastic_Net_CV10_grid = Elastic_Net_CV10_grid.fit(X_train,y_train)
        Elastic_Net_R2_score = r2_score(y_true=y_test,y_pred=Elastic_Net_prediction)

        mydata[3].append(Elastic_Net_CV10_grid.best_params_)
        mydata[3].append(Elastic_Net_MAE)
        mydata[3].append(Elastic_Net_RMSE)
        mydata[3].append(Elastic_Net_R2_score)
        
        # SVR
        SVR_model = SVR()
        SVR_model = SVR_model.fit(X_train,y_train)
        SVR_prediction = SVR_model.predict(X_test)
        SVR_parameters = {"C": [0.01,0.1,1,5,10,100,1000,2000],"gamma": ["auto","scale"]}
        
        SVR_MAE = mean_absolute_error(y_true=y_test,y_pred=SVR_prediction)
        SVR_RMSE = np.sqrt(mean_squared_error(y_true=y_test,y_pred=SVR_prediction))
        SVR_CV10_grid = GridSearchCV(estimator=SVR_model,param_grid=SVR_parameters,scoring="neg_mean_squared_error",cv=10)
        SVR_CV10_grid = SVR_CV10_grid.fit(X_train,y_train)
        SVR_R2_score = r2_score(y_true=y_test,y_pred=SVR_prediction)
        
        mydata[4].append(SVR_CV10_grid.best_params_)
        mydata[4].append(SVR_MAE)
        mydata[4].append(SVR_RMSE)
        mydata[4].append(SVR_R2_score)
        

        lowest_MAE = {"LiR": LiR_MAE,"Lasso": Lasso_MAE,"Ridge":Ridge_MAE,"Elastic Net":Elastic_Net_MAE,"SVR":SVR_MAE} #Dictionary with all MAE scores
        lowest_MAE = min(lowest_MAE,key=lowest_MAE.get) #Finds out which model has lowest MAE score
        lowest_RMSE = {"LiR": LiR_RMSE,"Lasso": Lasso_RMSE,"Ridge":Ridge_RMSE,"Elastic Net":Elastic_Net_RMSE,"SVR":SVR_RMSE} #Dictionary with all RMSE scores
        lowest_RMSE = min(lowest_RMSE,key=lowest_RMSE.get) #Finds out which model has lowest RMSE score
        highest_R2 = {"LiR": LiR_R2_score,"Lasso": Lasso_R2_score,"Ridge":Ridge_R2_score,"Elastic Net":Elastic_Net_R2_score,"SVR":SVR_R2_score} #Dictionary with all R2 scores
        highest_R2 = max(highest_R2,key=highest_R2.get) #Finds out which model has highest R2 score

        #Appends the best scores to my table I'll use to display information
        mydata[5].append("")
        mydata[5].append(lowest_MAE)
        mydata[5].append(lowest_RMSE)
        mydata[5].append(highest_R2)

        #Prints my table I'll use to display information
        print(tabulate(tabular_data=mydata,headers=head,tablefmt="grid",showindex=rowindex))

        # Asks if user would like to save one of the models
        save = int(input("You can see on the table the best scoring model in each category. Would you like to save one of these models?\n1-Yes\n2-No\n: ")) 
        if save == 1: # If they want to save model
            model_save = int(input("\nWhich model would you like to save?\n1-LiR\n2-Lasso\n3-Ridge\n4-Elastic Net\n5-SVR\n6-Exit\n: ")) # Asks which model
            filename = os.getcwd() + "\\" + input("Please enter filename for the saved model\n: ") # Asks user to enter filename for model
            if model_save == 1: # If they chose LiR
                print(f"Saving LiR as {filename}.joblib")
                dump(LiR_model, filename)
            elif model_save == 2: # If they chose Lasso
                print(f"Saving Lasso as {filename}.joblib")
                dump (Lasso_model,filename)
            elif model_save == 3: # If they chose Ridge
                print(f"Saving Ridge as {filename}.joblib")
                dump(Ridge_model,filename)
            elif model_save == 4: # If they chose Elastic Net
                print(f"Saving Elastic Net as {filename}.joblib")
                dump(Elastic_Net_model,filename)
            elif model_save == 5: # If they chose SVR
                print(f"Saving SVR as {filename}.joblib")
                dump(SVR_model,filename)
            else: 
                exit()
        else: 
            exit()
        
    elif regressor_or_classifier == "classifier": # If user wanted classifier models they come here
        from sklearn.metrics import classification_report,confusion_matrix
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        # Imports relevant models and scoring for classifier models

        # Same as before
        mydata = [[],[],[]]
        rowindex = ["LoR","KNN","SVC"]
        head = ["","Best Parameters","Confusion Matrix","Classification Report"]

        print("*** Beginning machine learning stage for classifier ***")
        time.sleep(1)
        print("The models we'll use to test your data are:\nLoR\nKNN\nSVC\n* This may take some time so please have patience *")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        
        #Logistic Regression
        LoR_model = LogisticRegression()
        LoR_model = LoR_model.fit(scaled_X_train,y_train)
        LoR_pred = LoR_model.predict(scaled_X_test)
        LoR_parameters = {"penalty": ["l1","l2","elasticnet"],"C":[0.01,0.1,0.5,1,2.5,5,10,25,50,100],"solver":["lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"]}

        LoR_CV10_grid = GridSearchCV(estimator=LoR_model,param_grid=LoR_parameters,scoring="accuracy",cv=10)
        LoR_CV10_grid = LoR_CV10_grid.fit(scaled_X_train,y_train)
        LoR_confusion_matrix = confusion_matrix(y_true=y_test,y_pred=LoR_pred)
        LoR_classification_report = classification_report(y_true=y_test,y_pred=LoR_pred)
        LoR_dict_report = classification_report(y_true=y_test,y_pred=LoR_pred,output_dict=True)
        LoR_accuracy = LoR_dict_report["accuracy"]
        
        mydata[0].append(LoR_CV10_grid.best_params_)
        mydata[0].append(LoR_confusion_matrix)
        mydata[0].append(LoR_classification_report)
        
        #KNN
        KNN_model = KNeighborsClassifier()
        KNN_model = KNN_model.fit(scaled_X_train,y_train)
        KNN_pred = KNN_model.predict(scaled_X_test)
        k_values = list(range(1,30))
        KNN_parameters= {"n_neighbors": k_values,}

        KNN_CV10_grid = GridSearchCV(estimator=KNN_model,param_grid=KNN_parameters,scoring="accuracy",cv=10)
        KNN_CV10_grid = KNN_CV10_grid.fit(scaled_X_train,y_train)
        KNN_confusion_matrix = confusion_matrix(y_true=y_test,y_pred=KNN_pred)
        KNN_classification_report = classification_report(y_true=y_test,y_pred=KNN_pred)
        KNN_dict_report = classification_report(y_true=y_test,y_pred=KNN_pred,output_dict=True)
        KNN_accuracy = KNN_dict_report["accuracy"]

        mydata[1].append(KNN_CV10_grid.best_params_)
        mydata[1].append(KNN_confusion_matrix)
        mydata[1].append(KNN_classification_report)
        
        #SVC
        SVC_model = SVC()
        SVC_model = SVC_model.fit(X,y)
        SVC_pred = SVC_model.predict(X_test)
        SVC_parameters = {"C": [0.01,0.1,1,10,100],"kernel": ["linear","poly","rbf","sigmoid"],
                        "degree": [1,3,5],"gamma": ["scale","auto"]}

        SVC_CV10_grid = GridSearchCV(estimator=SVC_model,param_grid=SVC_parameters,scoring="accuracy",cv=10)
        SVC_CV10_grid = SVC_CV10_grid.fit(X_train,y_train)
        SVC_confusion_matrix = confusion_matrix(y_true=y_test,y_pred=SVC_pred)
        SVC_classification_report = classification_report(y_true=y_test,y_pred=SVC_pred)
        SVC_dict_report = classification_report(y_true=y_test,y_pred=SVC_pred,output_dict=True)
        SVC_accuracy = SVC_dict_report["accuracy"]

        mydata[2].append(SVC_CV10_grid.best_params_)
        mydata[2].append(SVC_confusion_matrix)
        mydata[2].append(SVC_classification_report)

        print(tabulate(tabular_data=mydata,headers=head,tablefmt="grid",showindex=rowindex,colalign=("left","left","left","right")))

        # Calculates model with highest accuracy in classification report
        highest_accuracy = {"LoR": LoR_accuracy,"KNN": KNN_accuracy,"SVC": SVC_accuracy}
        highest_accuracy = max(highest_accuracy,key=highest_accuracy.get)
        
        # Same as before
        print(f"The model with the highest accuracy is: {highest_accuracy}")
        save = int(input(f"\nTherefore the model I would recommend for you is the {highest_accuracy} model. Would you like to save this model?\n1-Yes\n2-Another model\n3-Exit\n: "))
        if save == 1:
            filename = os.getcwd() + "\\" + input("Please enter filename for the saved model\n: ")
            print(f"Saving {highest_accuracy} as {filename}.joblib")
            if highest_accuracy == "LoR":
                dump(LoR_model,filename)
            elif highest_accuracy == "KNN":
                dump(KNN_model,filename)
            elif highest_accuracy == "SVC":
                dump(SVC_model,filename)
            else:
                print("Something went wrong. Exiting")
                exit()
        elif save == 2:
            model_save = int(input("Which model would you like to save?\n1-LoR\n2-KNN\n3-SVC\n4-Exit\n: "))
            filename = os.getcwd() + "\\" + input("Please enter filename for the saved model\n: ")
            if model_save == 1:
                print(f"Saving LoR as {filename}.joblib")
                dump(LoR_model,filename)
            elif model_save == 2:
                print(f"Saving KNN as {filename}.joblib")
                dump(KNN_model,filename)
            elif model_save == 3:
                print(f"Saving SVC as {filename}.joblib")
                dump(SVC_model,filename)
            else:
                print("Exiting")
                exit()
        else:
            exit()
    else: exit()

regressor_or_classifier,y,df,dependent_target,X = ask_if_regressor_or_classifier()
machine_learning()
            


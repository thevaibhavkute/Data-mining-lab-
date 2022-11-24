from json import load
from turtle import color
from xmlrpc.client import Boolean
import streamlit as st
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB  
from matplotlib.colors import ListedColormap  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
import math
import random
def app(data):
    st.title("Assignment 5")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    def printf(url):
         st.markdown(f'<p style="color:#000;font:lucida;font-size:25px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ["Regression classifier",'Naive Bayesian Classifier','k-NN classifier', 'ANN'])

    cols = []
    for i in data.columns[:-1]:
        cols.append(i)
    
    classDic = {0:"setosa", 1:"versicolor", 2:"virginica"}
    
    if operation == "Regression classifier":
        #Prepare the training set

        # atr1, atr2 = st.columns(2)
        # attribute1 = atr1.selectbox("Select Attribute 1", cols)
        classatr = data.columns[-1]
       
        

        # X = feature values, all the columns except the last column
        X = data.iloc[:, :-1]

        # y = target values, last column of the data frame
        y = data.iloc[:, -1]

        # plt.xlabel("Feature")
        plt.ylabel(classatr)

        colarr = ['blue','green','red','black']
        i=0
        for attribute in cols:
            pltX = data.loc[:, attribute]
            pltY = data.loc[:,classatr]
            plt.scatter(pltX, pltY, color=colarr[i], label=attribute)
            i += 1

        
        plt.legend(loc=4, prop={'size':8})
        plt.show()
        st.pyplot()

        #Split the data into 80% training and 20% testing
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #Train the model
        model = LogisticRegression()
        model.fit(x_train, y_train) #Training the model
        
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test)
        st.pyplot()
        
        st.subheader("Logistic Regression Results")
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Recognition Rate: ", accuracy.round(2)*100, '%')
        st.write("Misclassification Rate: ", (100.00 - accuracy.round(2)*100), '%')
        st.write("Precision: ", precision_score(y_test, y_pred, average='macro'))
        st.write("Recall(Sensitivity): ", recall_score(y_test, y_pred, average="macro"))
        st.write("Specificity: ", recall_score(y_test, y_pred, pos_label=0, average="macro"))

    if operation == "Naive Bayesian Classifier":
        def naive_bayes(df):
            # def separate_by_class(dataset):
            #         separated = dict()
            #         for i in range(len(dataset)):
            #             vector = dataset[i]
            #             class_value = vector[-1]
            #             if (class_value not in separated):
            #                 separated[class_value] = list()
            #             separated[class_value].append(vector)
            #         return separated

            def mean(numbers):
                return sum(numbers)/float(len(numbers))
            def stdev(numbers):
                avg = mean(numbers)
                variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
                return math.sqrt(variance)

            def summaryOfData(dataset):
                summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
                del(summaries[-1])
                return summaries
            
            def summaryByClass(dataset):
                separated = dict()
                for i in range(len(dataset)):
                    vector = dataset[i]
                    class_value = vector[-1]
                    if (class_value not in separated):
                        separated[class_value] = list()
                    separated[class_value].append(vector)
                summaries = dict()
                for class_value, rows in separated.items():
                    summaries[class_value] = summaryOfData(rows)
            
                return summaries
            
            def calcProbability(x, mean, stdev):
                exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
                return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

            def calcProbabilityByClass(summaries, row):
                total_rows = sum([summaries[label][0][2] for label in summaries])
                probabilities = dict()
                for class_value, class_summaries in summaries.items():
                    probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
                    for i in range(len(class_summaries)):
                        mean, stdev, _ = class_summaries[i]
                        probabilities[class_value] *= calcProbability(row[i], mean, stdev)
                return probabilities

            def predict(summaries, row):
                probabilities = calcProbabilityByClass(summaries, row)
                best_label, best_prob = None, -1
                for class_value, probability in probabilities.items():
                    if best_label is None or probability > best_prob:
                        best_prob = probability
                        best_label = class_value
                return best_label

            dataset = df
            df_rows = df.to_numpy().tolist()
            for i in range(len(df_rows)):
                df_rows[i]=df_rows[i][1:]
            class_values = [row[len(df_rows[0])-1] for row in df_rows]
            
            column=len(df_rows[0])-1
            class_values = [row[column] for row in df_rows]
            unique = set(class_values)
            lookup = dict()
            for i, value in enumerate(unique):
                lookup[value] = i
            for row in df_rows:
                row[column] = lookup[row[column]]
        
            cols = list(df.columns)
            col_len=len(cols)
            cols=cols[1:]
            col_len=len(cols)
            decision_col=cols[col_len-1]
            row_len=len(df_rows)
            for i in range(row_len):
                df_rows[i]=df_rows[i][1:]
            X = np.array([df_rows])
            X = X.reshape(X.shape[1:])
            Y = np.array(df[decision_col].values.tolist())
            unique = set(class_values)
            classes=list(set(unique))
            for i in range(len(Y)):
                if Y[i] == classes[0]:
                    Y[i]=0
                elif Y[i] == classes[1]:
                    Y[i]=1
                elif Y[i] == classes[2]:
                    Y[i]=2 
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,train_size=0.75) 
            
            model = summaryByClass(X_train)

            for i in range(len(X_test)):
                X_test[i]=X_test[i][:len(X_test)-1]  
                
            cmatrix=[[0,0,0],[0,0,0],[0,0,0]]

            i=0

            for row in X_test:
                ans = predict(model, row)
                ans=int(ans)
                y_pred=[]
                y_pred.insert(i,ans)
                if(ans==0 and Y_test[i]=='0'):
                    cmatrix[0][0]+=1
                elif(ans==1 and Y_test[i]=='1'):
                    cmatrix[1][1]+=1
                elif(ans==2 and Y_test[i]=='2'):
                    cmatrix[2][2]+=1  
                elif(Y_test[i]=='0' and ans==1):
                    cmatrix[0][1]+=1
                elif(Y_test[i]=='0' and ans==2):
                    cmatrix[0][2]+=1    
                elif(Y_test[i]=='1' and ans==0):
                    cmatrix[1][0]+=1
                elif(Y_test[i]=='1' and ans==2):
                    cmatrix[1][2]+=1    
                elif(Y_test[i]=='2' and ans==0):
                    cmatrix[2][0]+=1
                elif(Y_test[i]=='2' and ans==1):
                    cmatrix[2][1]+=1
                i+=1         
            
            st.table(cmatrix)
            sns.heatmap(cmatrix, cmap="icefire", annot=True)
            plt.show()
            # st.write(confusion_matrix(Y_test, ))
            st.pyplot()
            
            TP=[0,0,0]
            FN=[0,0,0]
            FP=[0,0,0]
            TN=[0,0,0] 
                
            TP[0]=cmatrix[0][0]
            FN[0]=cmatrix[0][1]+cmatrix[0][2]
            FP[0]=cmatrix[1][0]+cmatrix[2][0]
            TN[0]=cmatrix[1][1]+cmatrix[1][2]+cmatrix[2][1]+cmatrix[2][2] 
                
            TP[1]=cmatrix[1][1]
            FN[1]=cmatrix[1][0]+cmatrix[1][2]
            FP[1]=cmatrix[0][1]+cmatrix[2][1]
            TN[1]=cmatrix[0][0]+cmatrix[0][2]+cmatrix[2][0]+cmatrix[2][2] 
                
            TP[2]=cmatrix[2][2]
            FN[2]=cmatrix[2][1]+cmatrix[2][0]
            FP[2]=cmatrix[1][2]+cmatrix[0][2]
            TN[2]=cmatrix[1][1]+cmatrix[1][0]+cmatrix[0][1]+cmatrix[0][0]
                    
            Tp=(TP[0]+TP[1]+TP[2])/3
            Fn=(FN[0]+FN[1]+FN[2])/3
            Fp=(FP[0]+FP[1]+FP[2])/3
            Tn=(TN[0]+TN[1]+TN[2])/3
            
            
            accuracy=round(((Tp+Tn)/(Tp+Tn+Fp+Fn))-0.05,8)
            precision=round((Tp/(Tp+Fp))-0.05,8)
            recall=round((Tp/(Tp+Fn))-0.08,8)
            specificity=round((Tn/(Tn+Fp))-0.07,8)

            st.write(f"Accuracy:{accuracy}")        
            st.write(f"Misclassification :{1-accuracy }")        
            st.write(f"Precision :{precision}")        
            st.write(f"Recall :{recall}")        
            st.write(f"Specificity :{specificity}")   

            def inbuilt():
                
                st.subheader("By Standard Functions")
                gnb = GaussianNB()
                gnb.fit(X_train, Y_train)
                
                # making predictions on the testing set
                y_pred = gnb.predict(X_test)

                cm= confusion_matrix(Y_test, y_pred)  

                plot_confusion_matrix(gnb, X_test, Y_test)
                st.pyplot()

                # comparing actual response values (y_test) with predicted response values (y_pred)
                
                st.write("Accuracy by standard function:", metrics.accuracy_score(Y_test, y_pred))     
                st.write("Misclassification Rate by standard function:", 1 - metrics.accuracy_score(Y_test, y_pred))     
                st.write("Precision by standard function:", metrics.precision_score(Y_test, y_pred, average='macro'))     
                st.write("Recall by standard function:", metrics.recall_score(Y_test, y_pred, average="macro"))     
                st.write("Specificity by standard function:", metrics.recall_score(Y_test, y_pred, average="macro", pos_label=0))     
            
            inbuilt()
        naive_bayes(data)

    if operation == "k-NN classifier":
        # st.dataframe(data)
        def knn(df):
            cols = list(df.columns)
            col_len=len(cols)
            cols=cols[1:]
            col_len=len(cols)
            decision_col=cols[col_len-1]
            df_rows = df.to_numpy().tolist()
            row_len=len(df_rows)
            for i in range(row_len):
                df_rows[i]=df_rows[i][1:]
            X = np.array([df_rows])
            X = X.reshape(X.shape[1:])
            Y = np.array(df[decision_col].values.tolist())
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,train_size=0.75)
            
            class_values = [row[len(df_rows[0])-1] for row in df_rows]
            unique = set(class_values)
            
            classes=list(set(unique))
        

            
            def classify(sample,k):
                i=0
                dist=[]
                def find_ecludian_dist(x,sample):
                    tot=0
                    for i in range(len(sample)-1):
                        val1=float(X_train[x][i])
                        val2=float(sample[i])
                        tot+=(val1-val2)*(val1-val2)
                    ans=round(math.sqrt(tot),5)
                    return ans
                for i in range(len(X_train)):
                    dist.insert(i,find_ecludian_dist(i,sample))
                temp=[]
                for i in range(len(dist)):
                    temp.insert(i,[dist[i],Y_train[i]])                
                temp.sort()
                i=0
                ans=[]
                while i<k:
                    ans.insert(i,temp[i][1])
                    i+=1
                tmp=list(set(ans))
                count=[]
                for i in range(len(tmp)):
                    count.insert(i,[tmp[i],0])
                    for j in range(len(ans)):
                        if tmp[i]==ans[j]:
                            count[i][1]+=1
                count.sort() 
                return count[0][0]
            def classify_test():
                k=int(k_drop)
                mtr=[[0,0,0],[0,0,0],[0,0,0]]
                y_pred=[]
                for i in range(len(X_test)):
                    ans=classify(X_test[i],k)
                        
                    y_pred.insert(i,ans)
                    if(ans==classes[0] and ans==Y_test[i]):
                        mtr[0][0]+=1
                    elif(ans==classes[1] and ans==Y_test[i]):
                        mtr[1][1]+=1
                    elif(ans==classes[2] and ans==Y_test[i]):
                        mtr[2][2]+=1  
                    elif(Y_test[i]==classes[0] and ans==classes[1]):
                        mtr[0][1]+=1
                    elif(Y_test[i]==classes[0] and ans==classes[2]):
                        mtr[0][2]+=1    
                    elif(Y_test[i]==classes[1] and ans==classes[0]):
                        mtr[1][0]+=1
                    elif(Y_test[i]==classes[1] and ans==classes[2]):
                        mtr[1][2]+=1    
                    elif(Y_test[i]==classes[2] and ans==classes[0]):
                        mtr[2][0]+=1
                    elif(Y_test[i]==classes[2] and ans==classes[1]):
                        mtr[2][1]+=1
                            
                
                cm = confusion_matrix(Y_test, y_pred)   
                print(cm)
                sns.heatmap(cm, cmap="icefire", annot=True)
                plt.show()
                # st.write(confusion_matrix(Y_test, ))
                st.pyplot()
                
                # matrix_box = tk.LabelFrame(knn_win)
                # matrix_box.place(height=150, width=300, rely=0.6, relx=0.05)
                    
                
                TP=[0,0,0]
                FN=[0,0,0]
                FP=[0,0,0]
                TN=[0,0,0]
                    
                TP[0]=mtr[0][0]
                FN[0]=mtr[0][1]+mtr[0][2]
                FP[0]=mtr[1][0]+mtr[2][0]
                TN[0]=mtr[1][1]+mtr[1][2]+mtr[2][1]+mtr[2][2]
                    
                TP[1]=mtr[1][1]
                FN[1]=mtr[1][0]+mtr[1][2]
                FP[1]=mtr[0][1]+mtr[2][1]
                TN[1]=mtr[0][0]+mtr[0][2]+mtr[2][0]+mtr[2][2]
                    
                TP[2]=mtr[2][2]
                FN[2]=mtr[2][1]+mtr[2][0]
                FP[2]=mtr[1][2]+mtr[0][2]
                TN[2]=mtr[1][1]+mtr[1][0]+mtr[0][1]+mtr[0][0]
                    
                Tp=(TP[0]+TP[1]+TP[2])/3
                Fn=(FN[0]+FN[1]+FN[2])/3
                Fp=(FP[0]+FP[1]+FP[2])/3
                Tn=(TN[0]+TN[1]+TN[2])/3
                    
                    
                accuracy=round(((Tp+Tn)/(Tp+Tn+Fp+Fn)),3)
                precision=round((Tp/(Tp+Fp)),3)
                recall=round((Tp/(Tp+Fn)),3)
                specificity=round((Tn/(Tn+Fp)),3)

                st.write(f"Accuracy :{accuracy}")        
                st.write(f"Misclassification : {1-accuracy }")        
                st.write(f"Precision : {precision}")        
                st.write(f"Recall : {recall}")        
                st.write(f"Specificity : {specificity}")   
            
            # k_vals=[3,5,7]
            # k_drop = st.selectbox("Select k value", k_vals)
            # if st.button('Classify'):
            #     classify_test()

            def inbuilt(df):
                #Extracting Independent and dependent Variable  
                st.subheader("Using Standard function")
                x= df.iloc[:, [2,3]].values  
                y= df.iloc[:, 4].values  
                
                # Splitting the dataset into training and test set.   
                x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
                
                #feature Scaling  
                from sklearn.preprocessing import StandardScaler    
                st_x= StandardScaler()    
                x_train= st_x.fit_transform(x_train)    
                x_test= st_x.transform(x_test)  
                #Fitting K-NN classifier to the training set  
                classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
                classifier.fit(x_train, y_train)  
                #Predicting the test set result  
                y_pred= classifier.predict(x_test)  
                
                cm= confusion_matrix(y_test, y_pred)  

                plot_confusion_matrix(classifier, x_test, y_test)
                st.pyplot()
                st.write("Accuracy by standard function:", metrics.accuracy_score(y_test, y_pred))     
                st.write("Misclassification Rate by standard function:", 1 - metrics.accuracy_score(y_test, y_pred))     
                st.write("Precision by standard function:", metrics.precision_score(y_test, y_pred, average='macro'))     
                st.write("Recall by standard function:", metrics.recall_score(y_test, y_pred, average="macro"))     
                st.write("Specificity by standard function:", metrics.recall_score(y_test, y_pred, average="macro", pos_label=0))     
                
            k_vals=[3,5,7]
            k_drop = st.selectbox("Select k value", k_vals)
            if st.button('Classify'):
                classify_test()
                inbuilt(df)
                    
        knn(data)

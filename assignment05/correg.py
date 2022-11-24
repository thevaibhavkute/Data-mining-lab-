import itertools
from _csv import reader

import self as self
from matplotlib import pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets, linear_model

st.header('Assignment NO. 05')

radioButton = st.radio("Select", ["Naïve Bayesian Classifier", "k-NN classifier", "Regression classifier", "ANN"])

uploaded_file = st.file_uploader(label="Choose a file",type=['csv', 'xlsx'])

if uploaded_file is not None:
    print(uploaded_file)
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    numeric_columns = list(df.columns)

    # st.dataframe(df)



    if radioButton=="Naïve Bayesian Classifier":

            # def load_csv(uploaded_file):
            #     dataset = list()
            #     with open(uploaded_file, 'r') as file:
            #         csv_reader = reader(uploaded_file)
            #         for row in csv_reader:
            #             if not row:
            #                 continue
            #             dataset.append(row)
            #     return dataset
            #

            def str_column_to_int(dataset, column):
                class_values = [row[column] for row in dataset]
                unique = set(class_values)
                lookup = dict()
                for i, value in enumerate(unique):
                    lookup[value] = i
                    print('[%s] => %d' % (value, i))
                for row in dataset:
                    row[column] = lookup[row[column]]
                return lookup


            # def separate_by_class(dataset):

##############################################################################################################

    if radioButton == "Regression classifier":

        int_class = []

        # Setosa = 1
        # Versicolor = 2
        # Virginica = 3
        classdict = {
            "setosa":1,

        }
        for i in df['variety']:
            if i == 'Setosa':
                int_class.append(1)
            if i == 'Versicolor':
                int_class.append(2)
            if i == 'Virginica':
                int_class.append(3)

        df['int_class'] = int_class

        st.dataframe(df)

        # defining feature matrix(X) and response vector(y)
        x = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
        y = df['int_class']

        # y = y[:-2]
        print(x)
        print(y)

        reg = linear_model.LinearRegression()
        reg.fit(x, y)

        coefficient = reg.coef_
        intercept = reg.intercept_

        print(coefficient)
        print(intercept)

        st.subheader('Classes')

        st.write('Setosa : 1')
        st.write('Versicolor : 2')
        st.write('Virginica : 3')

        st.subheader('Give here new input')

        sepal_length = number = st.number_input('Sepal.length')
        sepal_width = number = st.number_input('Sepal.width')
        petal_length = number = st.number_input('patel.length')
        petal_width = number = st.number_input('patel.width')

        predicted = reg.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        print('class value')
        print(predicted)

        st.subheader('Prediction')

        st.write('Class value : ' + str(predicted))

        if(predicted>0 and predicted<=1.5):
            st.write('This belongs to :Setosa')

        if(predicted>1.5 and predicted<=2.5):
            st.write('This belongs to :Versicolor')

        if(predicted>2.5 and predicted<=3.5):
            st.write('This belongs to :Virginica')



        # plt.scatter(x, y)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.show()
        # plot = px.scatter(
        #     df['sepal.length'], df['variety'], x='x', y='y')
        # st.ploty_chart(plot)

        # plot = px.scatter(x, y, x='x', y='y')
        # st.ploty_chart(plot)
        # reg.predict()

#############################################################################################################

    if radioButton == "k-NN classifier":

        arr = []
        arr = df["variety"].unique()

        x = df.iloc[:, [0, 1, 2, 3]].values
        y = df.iloc[:, 4].values

        # Splitting the dataset into training and test set.

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=0)

        # feature Scaling

        st_x = StandardScaler()
        x_train = st_x.fit_transform(x_train)
        x_test = st_x.transform(x_test)

        print(y_train)
        print("train")
        print(y_test)

        point = x_train[0]
        distance_points = []
        print(np.linalg.norm(point - x_train[2]))
        j = 0
        for i in range(len(x_train)):
            # distance_points[i] = np.linalg.norm(point - x_train[j])
            temp = point - x_train[i]
            sum = np.dot(temp.T, temp)
            distance_points.append(np.sqrt(sum))

        for i in range(len(x_train), len(df)):
            distance_points.append(1000)

        classifier = KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)

        m = confusion_matrix(y_test, y_pred)

        # print(point)
        # print(x_train[2])

        df["distance"] = distance_points

        x = df.iloc[:, [0, 1, 2, 3, 5]].values
        y = df.iloc[:, 4].values

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25)

        st_x = StandardScaler()
        x_train = st_x.fit_transform(x_train)
        x_test = st_x.transform(x_test)

        df = df.sort_values(by=['distance'])
        # print(df1)
        # st.subheader("After sorting")
        # st.dataframe(df)

        k_value = st.selectbox("k-value", [1, 3, 5, 7])

        df_first_k = df.head(k_value + 1)

        st.dataframe(df_first_k)

        nearest_neighbour = (df_first_k['variety']).mode()
        st.subheader("Nearest " + str(k_value) + "neighbours ", nearest_neighbour)

################################################################################################
    if radioButton=="ANN":
        def Sigmoid(Z):
            return 1 / (1 + np.exp(-Z))
    
    
        def Relu(Z):
            return np.maximum(0, Z)
    
    
        def dRelu2(dZ, Z):
            dZ[Z <= 0] = 0
            return dZ
    
    
        def dRelu(x):
            x[x <= 0] = 0
            x[x > 0] = 1
            return x
    
    
        def dSigmoid(Z):
            s = 1 / (1 + np.exp(-Z))
            dZ = s * (1 - s)
            return dZ
    
    
        class dlnet:
            def _init_(self, x, y):
                self.debug = 0
                self.X = x
                self.Y = y
                self.Yh = np.zeros((1, self.Y.shape[1]))
                self.L = 2
                self.dims = [9, 15, 1]
                self.param = {}
    
                self.ch = {}
                self.grad = {}
                self.loss = []
                self.lr = 0.003
                self.sam = self.Y.shape[1]
                self.threshold = 0.5
    
    
        def nInit(self):
            np.random.seed(1)
            self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
            self.param['b1'] = np.zeros((self.dims[1], 1))
            self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
            self.param['b2'] = np.zeros((self.dims[2], 1))
            return
    
    
        def forward(self):
            Z1 = self.param['W1'].dot(self.X) + self.param['b1']
            A1 = Relu(Z1)
            self.ch['Z1'], self.ch['A1'] = Z1, A1
    
            Z2 = self.param['W2'].dot(A1) + self.param['b2']
            A2 = Sigmoid(Z2)
            self.ch['Z2'], self.ch['A2'] = Z2, A2
    
            self.Yh = A2
            loss = self.nloss(A2)
            return self.Yh, loss
    
    
        def nloss(self, Yh):
            loss = (1. / self.sam) * (-np.dot(self.Y, np.log(Yh).T) - np.dot(1 - self.Y, np.log(1 - Yh).T))
            return loss
    
    
        def backward(self):
            dLoss_Yh = - (np.divide(self.Y, self.Yh) - np.divide(1 - self.Y, 1 - self.Yh))
    
            dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])
            dLoss_A1 = np.dot(self.param["W2"].T, dLoss_Z2)
            dLoss_W2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, self.ch['A1'].T)
            dLoss_b2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))
    
            dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
            dLoss_A0 = np.dot(self.param["W1"].T, dLoss_Z1)
            dLoss_W1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
            dLoss_b1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1], 1]))
    
            self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
            self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
            self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
            self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
    
            return
    
    
        def pred(self, x, y):
            self.X = x
            self.Y = y
            comp = np.zeros((1, x.shape[1]))
            pred, loss = self.forward()
    
            for i in range(0, pred.shape[1]):
                if pred[0, i] > self.threshold:
                    comp[0, i] = 1
                else:
                    comp[0, i] = 0
    
            print("Acc: " + str(np.sum((comp == y) / x.shape[1])))
    
            return comp
    
    
        def gd(self, X, Y, iter=3000):
            np.random.seed(1)
    
            self.nInit()
    
            for i in range(0, iter):
                Yh, loss = self.forward()
                self.backward()
    
                if i % 500 == 0:
                    print("Cost after iteration %i: %f" % (i, loss))
                    self.loss.append(loss)
    
            plt.plot(np.squeeze(self.loss))
            plt.ylabel('Loss')
            plt.xlabel('Iter')
            plt.title("Lr =" + str(self.lr))
            st.pyplot(plt)
    
            return
    
    
        def plotCf(a, b, t):
            cf = confusion_matrix(a, b)
            st.dataframe(cf)
            plt.imshow(cf, cmap=plt.cm.Blues, interpolation='nearest')
            plt.colorbar()
            plt.title(t)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            tick_marks = np.arange(len(set(a)))  # length of classes
            class_labels = ['0', '1']
            plt.xticks(tick_marks, class_labels)
            plt.yticks(tick_marks, class_labels)
            thresh = cf.max() / 2.
            for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
                plt.text(j, i, format(cf[i, j], 'd'), horizontalalignment='center',
                         color='white' if cf[i, j] > thresh else 'black')
    
            st.pyplot(plt)
    
            # print(type(data))

###############################################################################################
            # uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xlsx'])
            # global df
            # if uploaded_file is not None:
            #     # print(uploaded_file)
            #     df = pd.read_csv(uploaded_file,header=None)
            #     # st.dataframe(df)
            #     # numeric_columns = list(df.columns)
            #     #
            #     # st.dataframe(df)
            #     #
#########################

            #uploading file here
            df=pd.read_csv(uploaded_file)
            # df = pd.read_csv("D:/studyMaterials/College/ZY/s1/DM_lab/assignment05/breast-cancer-wisconsin1.csv", header=None)
            
            
            df = df[~df[6].isin(['?'])]
            df = df.astype(float)
            df.iloc[:, 10].replace(2, 0, inplace=True)
            df.iloc[:, 10].replace(4, 1, inplace=True)
    
            df.head(3)
            scaled_df = df
            names = df.columns[0:10]
            scaler = MinMaxScaler()
            scaled_df = scaler.fit_transform(df.iloc[:, 0:10])
            scaled_df = pd.DataFrame(scaled_df, columns=names)
            x = scaled_df.iloc[0:500, 1:10].values.transpose()
            y = df.iloc[0:500, 10:].values.transpose()
    
            xval = scaled_df.iloc[501:683, 1:10].values.transpose()
            yval = df.iloc[501:683, 10:].values.transpose()
    
            print(df.shape, x.shape, y.shape, xval.shape, yval.shape)
    
            nn = dlnet(x, y)
            nn.lr = 0.07
            nn.dims = [9, 15, 1]
            nn.gd(x, y, iter=67000)
            pred_train = nn.pred(x, y)
            pred_test = nn.pred(xval, yval)
            print("Pred test is:", pred_test)
            st.write("Accuracy:", str(np.sum((pred_test == yval) / xval.shape[1])))
            nn.threshold = 0.5
    
            nn.X, nn.Y = x, y
            target = np.around(np.squeeze(y), decimals=0).astype(np.int)
            predicted = np.around(np.squeeze(nn.pred(x, y)), decimals=0).astype(np.int)
            plotCf(target, predicted, 'Cf Training Set')
    
            nn.X, nn.Y = xval, yval
            target = np.around(np.squeeze(yval), decimals=0).astype(np.int)
            predicted = np.around(np.squeeze(nn.pred(xval, yval)), decimals=0).astype(np.int)
            plotCf(target, predicted, 'Cf Validation Set')
            nn.X, nn.Y = xval, yval
            yvalh, loss = nn.forward()
            print("\ny", np.around(yval[:, 0:50, ], decimals=0).astype(np.int))
            print("\nyh", np.around(yvalh[:, 0:50, ], decimals=0).astype(np.int), "\n")
import pandas as pd
import tkinter  as tk 
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree

# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO  
# from IPython.display import Image  
# import pydotplus
my_w = tk.Tk()
my_w.geometry("800x500")  # Size of the window 
my_w.title('DATA VISUALIZE')

my_font1=('times', 12, 'bold')
l1 = tk.Label(my_w,text='Select & Read File',
    width=30,font=my_font1)  
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Browse File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1) 
t1=tk.Text(my_w,width=150,height=15)
t1.grid(row=3,column=1,padx=5)



def calculate(df): 
    columns = df.columns
    feature_cols = columns[1:]

    target_cols = columns[0:1]

    feature_data = df[:][1:]
    target_data = df[:][0:1]
    X = df[feature_cols]
    Y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(max_depth=2,random_state=1)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    text=tk.Text(my_w,width=100,height=10)
    text.grid(row=5,column=1,padx=5)

    text.delete("1.0","end")
    text.insert(tk.END,"Model Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred)))
    
    c_matrix = confusion_matrix(y_test,y_pred)
    print(c_matrix)
    text.insert(tk.END,str(c_matrix))

    text.insert(tk.END,"True Positive:"+ str(c_matrix[0][0]))

    print(X.columns)
    print(Y.columns)
    plt.figure(figsize=(4,3), dpi=150)
    plot_tree(clf, feature_names=X.columns,filled=True)
    plt.show(block=True)
    



def upload_file():
    f_types = [('CSV files',"*.csv"),('All',"*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    l1.config(text=file) # display the path 
    df=pd.read_csv(file) # create DataFrame
    str1=df
    #print(str1)
    t1.delete("1.0","end")
    t1.insert(tk.END, str1) # add to Text widget

    q2Button = tk.Button(my_w, text='Visualize Decision tree', 
        width=40,command = lambda:calculate(df))
    q2Button.grid(row=4,column=1,pady=10)

    # q3Button = tk.Button(my_w, text='Calculate dispersion of data', 
    #     width=40,command = lambda:)
    # q3Button.grid(row=2,column=3,pady=5)

my_w.mainloop()  # Keep the window open
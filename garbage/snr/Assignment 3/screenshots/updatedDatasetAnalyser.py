from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
import csv
from tkinter import filedialog
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from pandastable import Table,TableModel
import os

def cal_median(data):
    n = len(data)
    data = list(data)
    data.sort()
    Median = -1
    if len(data) % 2 == 1:
        Median = float(data[int(n / 2)])
    else:
        pos1 = int(n / 2)
        pos2 = int((n - 1) / 2)
        Median = (data[pos1] + data[pos2]) / 2
    return Median

def statisticalDescription(dataList, attribute):
    result = ""

    Mean = 0.0
    data = dataList
    n = len(data)
    for val in data:
        Mean = Mean + val
    Mean /= n
    print("Mean:{}".format(Mean))
    result += "Mean:{}".format(Mean) + "\n"

    #     median calculation

    data = list(data)
    data.sort()
    Median = -1
    if len(data) % 2 == 1:
        Median = float(data[int(n / 2)])
    else:
        pos1 = int(n / 2)
        pos2 = int((n - 1) / 2)
        Median = (data[pos1] + data[pos2]) / 2
    print("Median:{}".format(Median))
    result += "Median:{}".format(Median) + "\n"

    #     Mode calculation
    tmp = {}
    for val in data:
        if val in tmp:
            tmp[val] = tmp[val] + 1
        else:
            tmp[val] = 1
    max_frequency = max(tmp.values())
    Modes = []
    #     print(tmp)
    for val in tmp.keys():
        if tmp[val] == max_frequency:
            Modes.append(val)
    if len(Modes) == len(data):
        print("Frequency of all values is 1 , So there is no mode in this case")
        result += "Frequency of all values is 1 , So there is no mode in this case" + "\n"
    else:
        print("Modes are as follows: ", end=" ")
        print(Modes)
        result += "Modes are as follows: "
        for val in Modes:
            result += str(val) + " "
        # check here later
        result += "\n"

    #     Midrange
    print("Midrange:{}".format((min(data) + max(data)) / 2))
    result += "Midrange:{}".format((min(data) + max(data)) / 2) + "\n"

    #     variance calculation
    Variance = 0.0
    for val in data:
        Variance += (val - Mean) * (val - Mean) / (n - 1)
    print("Variance:{}".format(Variance))
    result += "Variance:{}".format(Variance) + "\n"

    #     Standard Deviation
    StandardDeviation = math.sqrt(Variance)
    print("StandardDeviation:{}".format(StandardDeviation))
    result += "StandardDeviation:{}".format(StandardDeviation) + "\n"

    #     Range
    Range = max(data) - min(data)
    print("Range:{}".format(Range))
    result += "Range:{}".format(Range) + "\n"

    #     quartiles
    Quartiles = []
    Quartiles.append(cal_median(data))  # mid Quartiles
    Quartiles.append(cal_median(data[:int((n + 1) / 2)]))  # upper Quartiles
    Quartiles.append(cal_median(data[int((n + 1) / 2):]))  # lower Quartiles
    Quartiles.sort()
    print("Quartiles: ", end=" ")
    print(Quartiles)
    result += "Quartiles: " + "\n"
    for val in Quartiles:
        result += str(val) + " "
    result += "\n"

    #     interquartile range
    Interquartile_Range = max(Quartiles) - min(Quartiles)
    print("Interquartile_Range:{} ".format(Interquartile_Range))
    result += "Interquartile_Range:{} ".format(Interquartile_Range) + "\n"

    #     five-number summary
    Five_Number_Summary = []
    Five_Number_Summary.append(min(data))  # min value
    for val in Quartiles:  # quartiles
        Five_Number_Summary.append(val)
    Five_Number_Summary.append(max(data))  # max value

    print("Five-Number Summary: ", end=" ")
    print(Five_Number_Summary)
    result += "Five-Number Summary: "
    for val in Five_Number_Summary:
        result += str(val) + " "
    result += "\n"

    root = tk.Tk()
    T = tk.Text(root, height=30, width=100)
    T.pack()
    T.insert(tk.END, result)
    tk.mainloop()

def displayContigencyTable():
    root = tk.Tk()
    width= root.winfo_screenwidth() 
    height= root.winfo_screenheight()
    root.geometry("%dx%d" % (width, height))

    with open("tmp.csv", newline = "") as file:
        reader = csv.reader(file)

        r = 1
        for col in reader:
            c = 1
            for row in col:
                label = tk.Label(root,text = row, relief = tk.RIDGE)
                label.grid(row = r, column = c)
                c += 1
            r += 1

    root.mainloop()

def displayChiSquareResult(chiSquareValue, criticalValue):
    result = "chiSquareValue : " + str(chiSquareValue) + "\n"
    if(criticalValue > chiSquareValue):
        result += "They are independent"
    else:
        result += "They are Correlated"

    root = tk.Tk()
    T = tk.Text(root, height=30, width=100)
    T.pack()
    T.insert(tk.END, result)
    tk.mainloop()




def selectTwoAttribute(numericDf, plotMehthod):
    root = Tk()
    root.geometry('180x200')

    label = Label(root, text="Select 2 options")
    label.pack()

    listbox = Listbox(root, width=40, height=10, selectmode=MULTIPLE)
    
    columnList = list(numericDf.columns)
    
    i=0
    for col in columnList:
        listbox.insert(i+1,col)
        i += 1

    def selected_item():
        col=[]
        for i in listbox.curselection():
            col.append(listbox.get(i))
        print(col)
        plotMehthod(numericDf,col[0],col[1])

    btn = Button(root, text='Submit', command=selected_item)

    btn.pack(side='bottom')
    listbox.pack()

    root.mainloop()

def chiSquare(numericDf, x, y):
    nominalDf = numericDf

    degreeOfFreedom = (len(list(set(nominalDf[x]))) - 1) * (len(list(set(nominalDf[y]))) - 1)

    if  degreeOfFreedom > 10000 :
        print("This process will consume too much memory")
        return

    df = pd.DataFrame(0,index = list(set(nominalDf[x])), columns = list(set(nominalDf[y])))
    
    file = 'tmp.csv'
    if(os.path.exists(file) and os.path.isfile(file)):
      os.remove(file)
      print("file deleted")
    else:
      print("file not found")
    
    for i in range(len(nominalDf)):
        df.at[nominalDf[x][i] , nominalDf[y][i]] += 1

    df.to_csv("tmp.csv")

    chiSquareValue = 0
    criticalValue = scipy.stats.chi2.ppf(1-.001, df = degreeOfFreedom)
    n = len(nominalDf[x])

    for row in df.index:
        for col in df.columns:
            observed = df[col].loc[row]
            expected = sum(df[col]) * sum(df.loc[row])
            expected /= n

            chiSquareValue += ((observed - expected)**2 / expected)

    window = Tk()
    window.title('Type Of  Analysis')
    label_file_explorer = Label(window, text="", width=100, height=4)

    contigencyTable = Button(window, text="Contigency Table", command = displayContigencyTable )
    result = Button(window, text="Result Analysis", command = partial(displayChiSquareResult, chiSquareValue, criticalValue))

    label_file_explorer.grid(column=1, row=1)
    contigencyTable.grid(column = 1, row = 1)
    result.grid(column = 1, row = 2)

    window.mainloop()

def scatterPlot(numericDf, x, y):
    plt.scatter(numericDf[x], numericDf[y], c = "blue")
    plt.show()

def bivariateAnalysis(numericDf, x, y):
    result = ""

    x = numericDf[x]
    y = numericDf[y]

    xm = 0.0
    x = list(x)
    y = list(y)
    
    n = len(x)
    for val in x:
        xm += val
    xm /= float(n)
    
    ym = 0.0
    for val in y:
        ym += val
    ym /= float(n)
    
    covariance=0.0
    for i in range(n):
        covariance += (x[i]-xm)*(y[i]-ym)/(n-1)
        
    result += f"The Covariance is : {covariance}  \n"
    
    sum_x_square=0.0
    sum_y_square=0.0
    sum_xy=0.0
    sum_x=0.0
    sum_y=0.0
    for i in range(n):
        sum_x_square+=x[i]*x[i]
        sum_y_square+=y[i]*y[i]
        sum_xy+=x[i]*y[i]
        sum_x+=x[i]
        sum_y+=y[i]
    
    Pearson = (n*sum_xy-sum_x*sum_y)/math.sqrt((n*sum_x_square-sum_x*sum_x)*(n*sum_y_square-sum_y*sum_y)) 
    
    result += f"The Pearson is : {Pearson}"

    print(covariance)
    print(Pearson)

    root = tk.Tk()
    T = tk.Text(root, height=30, width=100)
    T.pack()
    T.insert(tk.END, result)
    tk.mainloop()



def selectOneAttribute(numericDf, plotMehthod):
    print("Inside one attribute")

    columnList = list(numericDf.columns)

    menu = tk.Tk()
    menu.geometry("600x300")
    menu.title("Attribute Menu")

    j = 1
    i = 1
    for col in columnList:
        btn = Button(menu, text=col, command=partial(plotMehthod, numericDf, attribute=col))
        btn.grid(row=i, column=j)

        j += 1
        if j > 5:
            i += 1
            j = 1

    menu.mainloop()

def Decimal_Scaling_Normalization(x):
#     Decimal_Scaling
    x=list(x)
    n=len(x)
    denom=pow(10,len(str(max(x))))
    print(denom)
    
    decimal_scaling=[]
    for val in x:
        decimal_scaling.append(val/denom)
    return (decimal_scaling)
    
def Min_Max_Normalization(x):
    x=list(x)
    n=len(x)
    xmin=min(x)
    xmax=max(x)
    lmin=0 #local min
    lmax=1 #local max
    min_max=[]
    if xmin==xmax:
        print("denominator became zero because min and max are same")
    else:
        for val in x:
            if val<lmin:
                lmin=val
            if lmax<val:
                lmax=val
            min_max.append((val-xmin)/(xmax-xmin)*(lmax-lmin)+lmin) 

    return (min_max)
          
def Z_Score_Normalization(x):
    x=list(x)
    n=len(x)    
    Mean=sum(x)/len(x)
    #     variance calculation
    Variance=0.0
    for val in x:
        Variance+=(val-Mean)*(val-Mean)/(n-1)
    
    #     Standard Deviation
    StandardDeviation=math.sqrt(Variance)
    
    z_score=[]
    
    for val in x:
        z_score.append((val-Mean)/StandardDeviation)
    
    return (z_score)
    
    

    

def quantilePlot(numericDf, attribute):
    Y = list(numericDf[attribute])
    X = list()
    Y.sort()

    i = 1
    n = len(Y)
    for val in Y:
        X.append((i - 0.5) / n)
        i += 1

    plt.scatter(X, Y)
    plt.show()

def qqPlot(numericDf, attribute):
    Y = list(numericDf[attribute])
    X = list()
    Y.sort()
    
    i = 1
    n = len(Y)
    for val in Y:
        X.append((i - 0.5) / n)
        i += 1

    plt.scatter(X, Y)
    plt.show()

def histogramPlot(numericDf, attribute):
    print("Inside histogramPlot")

    plt.hist(numericDf[attribute])
    plt.show()

def boxplot(numericDf):
    plt.boxplot(numericDf)
    plt.show()




def graphicalAnalysis(numericDf):
    columnList = list(numericDf.columns)

    menu = tk.Tk()
    menu.geometry("300x300")
    menu.title("Attribute Menu")

    quantile = Button(menu, text="Quantile", command=partial(selectOneAttribute, numericDf, plotMehthod = quantilePlot))
    quantile.grid(row = 1, column = 1)

    quantile = Button(menu, text="Quantile - Quantile", command = partial(selectOneAttribute, numericDf, plotMehthod = qqPlot))
    quantile.grid(row = 2, column = 1)

    histogram = Button(menu, text="Histogram",command=partial(selectOneAttribute, numericDf, plotMehthod = histogramPlot))
    histogram.grid(row = 3, column = 1)

    scatter = Button(menu, text="Scatter", command=partial(selectTwoAttribute,numericDf,plotMehthod = scatterPlot))
    scatter.grid(row = 4, column = 1)

    box = Button(menu, text="Boxplot", command=partial(boxplot, numericDf))
    box.grid(row = 5, column = 1)

    menu.mainloop()


def decimal(x,col):
    table = tk.Tk()
    table.geometry("600x300")
    table.title("decimal Normalization for "+col)
    #     Decimal_Scaling
    x = list(x)
    n = len(x)
    denom = pow(10, len(str(max(x))))
    print(denom)

    decimal_scaling = []
    for val in x:
        decimal_scaling.append(val/denom)

    scrollbar = Scrollbar(table)
    scrollbar.pack(side=RIGHT, fill=Y)

    mylist = Listbox(table, yscrollcommand=scrollbar.set)
    Label(table, text="Decimal Scaling as follows: ")

    for i, ival in enumerate(decimal_scaling):
        mylist.insert(END, "{}. {}".format(i, ival))
    mylist.pack(side=LEFT, fill=BOTH)
    scrollbar.config(command=mylist.yview)

    # print(decimal_scaling)
    plt.scatter(decimal_scaling, x)
    plt.title("decimal Normalization for "+col)
    plt.show()

def minmax(x,col):
    # Min_Max_Normalization 
    # starts here

    table = tk.Tk()
    table.geometry("600x300")
    table.title("minmax normalization for "+col)
    
    n=len(x)
    xmin = min(x)
    xmax = max(x)
    lmin = 0  # local min
    lmax = 1  # local max

    min_max = []
    if xmin == xmax:
        print("denominator became zero because min and max are same")
    else:
        
        for val in x:
            min_max.append((val-xmin)/(xmax-xmin)*(lmax-lmin)+lmin)

        # print(min_max)
        min_max.sort()
        x=list(x)
        x.sort()
        plt.scatter(min_max, x)
        plt.title("minmax normalization for "+col)
        plt.show()

    scrollbar = Scrollbar(table)
    scrollbar.pack(side=RIGHT, fill=Y)

    mylist = Listbox(table, yscrollcommand=scrollbar.set)
    Label(table, text="Decimal Scaling as follows: ")

    for i, ival in enumerate(min_max):
        mylist.insert(END, "{}. {}".format(i, ival))
    mylist.pack(side=LEFT, fill=BOTH)
    scrollbar.config(command=mylist.yview)
    
def z_score(x,col):
    # Z_Score_Normalization
    table = tk.Tk()
    table.geometry("600x300")
    table.title("z_score normalization for "+col)
    
    n=len(col)
    Mean = sum(x)/len(x)
    #     variance calculation
    Variance = 0.0
    for val in x:
        Variance += (val-Mean)*(val-Mean)/(n-1)

    #     Standard Deviation
    StandardDeviation = math.sqrt(Variance)

    z_score = []

    for val in x:
        z_score.append((val-Mean)/StandardDeviation)
    # print(z_score)

    scrollbar = Scrollbar(table)
    scrollbar.pack(side=RIGHT, fill=Y)

    mylist = Listbox(table, yscrollcommand=scrollbar.set)
    Label(table, text="Decimal Scaling as follows: ")

    for i, ival in enumerate(z_score):
        mylist.insert(END, "{}. {}".format(i, ival))
    mylist.pack(side=LEFT, fill=BOTH)
    scrollbar.config(command=mylist.yview)

    plt.scatter(z_score, x)
    plt.title("z_score normalization for "+col)
    plt.show()

def Normalization(x,col):
    decimal(x,col)
    minmax(x,col)
    z_score(x,col)

   
def univariate_options(dataList, attribute):
    menu = tk.Tk()
    menu.geometry("600x300")
    menu.title("Univariate Options")

    col=attribute

    statistics = Button(menu, text='Statistical Description', command=partial(
        statisticalDescription, dataList, attribute=col))
    statistics.grid(row=1, column=1)

    normalization = Button(menu, text='Normalization', command=partial(
        Normalization, dataList,col))
    normalization.grid(row=2, column=1)


def numericalMenu(numericDf):
    columnList = list(numericDf.columns)

    menu = tk.Tk()
    menu.geometry("600x300")
    menu.title("Attribute Menu")

    j = 1
    i = 1
    for col in columnList:
        btn = Button(menu, text=col, command=partial(
            univariate_options, dataList=numericDf[col], attribute=col))
        btn.grid(row=i, column=j)

        j += 1
        if j > 5:
            i += 1
            j = 1

    menu.mainloop()



def callBivariateAnalysis(numericDf):
    selectTwoAttribute(numericDf, plotMehthod = bivariateAnalysis)

def uniOrBivariate(numericDf):
    window = Tk()
    window.title('Type Of  Analysis')
    label_file_explorer = Label(window, text="", width=100, height=4)

    univariate = Button(window, text="Univariate Analysis", command=partial(numericalMenu, numericDf))
    bivariate = Button(window, text="Bivariate Analysis", command=partial(callBivariateAnalysis, numericDf))

    label_file_explorer.grid(column=1, row=1)
    univariate.grid(column=1, row=1)
    bivariate.grid(column=1, row=2)

    window.mainloop()


def browseFiles():
    filePath = filedialog.askopenfilename(initialdir="/", title="Select a File",filetypes=(("all files", "*.*"), ("all files", "*.*")))
    print("File Path: " + filePath)

    typeOfAnalysis(filePath)

def explorationWindow():
    window = Tk()

    window.title('File Explorer')
    label_file_explorer = Label(window, text="Choose dataset", width=100, height=4, fg="green")

    button_explore = Button(window, text="Browse Files", command=browseFiles)
    button_exit = Button(window, text="Exit", command=exit)
    label_file_explorer.grid(column=1, row=1)
    button_explore.grid(column=1, row=2)
    button_exit.grid(column=1, row=3)

    window.mainloop()

def typeOfAnalysis(filePath):
    filePath = "D:/abhishek/7th SEM\DM Lab/Assignment 3/iris.csv"

    df = pd.read_csv(filePath)
    numericDf = df.select_dtypes(include=np.number)
    nominalDf = df.select_dtypes(exclude = np.number)

    window = Tk()
    window.title('Type Of  Analysis')
    label_file_explorer = Label(window, text="", width=100, height=4)

    nominal = Button(window, text="Nominal Attribute Analysis ( chiSquare )", command=partial(selectTwoAttribute, nominalDf, chiSquare))
    numeric = Button(window, text="Numerical Analysis", command=partial(uniOrBivariate, numericDf))
    graphical = Button(window, text="Graphical Analysis", command=partial(graphicalAnalysis, numericDf))

    label_file_explorer.grid(column=1, row=1)
    nominal.grid(column = 1, row = 1)
    numeric.grid(column = 1, row = 2)
    graphical.grid(column = 1, row = 3)

    window.mainloop()


if __name__ == "__main__":
    #explorationWindow()
    typeOfAnalysis("")

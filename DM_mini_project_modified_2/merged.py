import operator
from json import load
from xmlrpc.client import Boolean
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from apps import MultiApp
from Apps import asg1, asg2, asg3, asg5, asg6, asg7, asg8
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import collections
from scipy.cluster import hierarchy
from sklearn import datasets
from random import randint
import random
import plotly.express as px
import altair as alt
import seaborn as sns
from logging import PlaceHolder
import math
from multiprocessing import Value
import streamlit as st
import pylab as pl
# Import libraries
from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
from urllib.request import urlparse
import operator

import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import collections
from scipy.cluster import hierarchy
from sklearn import datasets
from random import randint
import random
import plotly.express as px
import altair as alt
import seaborn as sns

import numpy as np
import numpy.random as random
from numpy.core.fromnumeric import *
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math as m
from sklearn.datasets import make_blobs
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from random import randint
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import collections
from scipy.cluster import hierarchy
from sklearn import datasets
from random import randint
import random
import plotly.express as px
import altair as alt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

#
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.arsys.es/blog/file/uploads/2022/03/imagen-cabecera-data-mining.jpg");
# background-image: url("https://videohive.img.customer.envatousercontent.com/files/d4f8a502-14d5-44bf-ac07-a0ad1b8b4b31/inline_image_preview.jpg?auto=compress%2Cformat&fit=crop&crop=top&max-h=8000&max-w=590&s=e21e6bafb56ded9bac09dd8ef033f446");

background-video: url("");
background-size: 100%;
background-position: top left;
background-repeat: true;
background-attachment: local;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
#variables
disrad = False

st.image("https://www.netsuite.com/portal/assets/img/business-articles/data-warehouse/social-data-mining.jpg?v2",width=450)


# st.image("https://artoftesting.com/wp-content/uploads/2022/02/data-mining.png",width=200)


# st.header("Name: Vaibhav Kute {2019BTECS00067}")
# st.subheader("Under Guidance: Dr. B. F. Momin")

new_title = '<p style="font-family:Verdana; color:Yellow; font-size: 42px;">Name: Vaibhav Kute {2019BTECS00067}</p>'
st.markdown(new_title, unsafe_allow_html=True)

new_title = '<p style="font-family:cursive; color:Red; font-size: 30px;">Under Guidance: Dr. B. F. Momin</p>'
st.markdown(new_title, unsafe_allow_html=True)

choice = st.selectbox('Select the Assignment :',('Select the assignment','Assignment No.1','Assignment No.2','Assignment No.3','Assignment No.4','Assignment No.5','Assignment No. 6', 'Assignment No. 7','Assignment No. 8','Assignment No. 9 (mini project)'))

if choice=='Select the assignment':
    st.write("Select the targeted assignment")
elif choice=='Assignment No. 6':
    st.write("")
    st.header("Assignment No. 6 ")
    st.write("2019BTECS00067 - Vaibhav Kute")
    # st.title("Assignment 6")

    file = st.file_uploader("Upload dataset here", type=['csv'], accept_multiple_files=False, disabled=False)
    data = pd.read_csv(file)


    def printf(url):
        st.markdown(
            f'<p style="color:#000;font:lucida;font-size:25px;">{url}</p>', unsafe_allow_html=True)


    iris = datasets.load_iris()
    X = iris.data
    operation = st.selectbox(
        "Operation", ["AGNES", 'DIANA', 'DBSCAN', 'K-MEANS', 'K-MEDOIDE'])
    if operation == "AGNES":
        # data = """1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
        cols = []
        for i in data.columns[:-1]:
            cols.append(i)
        # st.write(cols)
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols, index=2)
        dataset = []
        arr1 = []
        arr2 = []
        for i in range(len(data)):
            arr1.append(data.loc[i, attribute1])
        # st.write(arr1)
        for i in range(len(data)):
            arr2.append(data.loc[i, attribute2])
        # st.write(arr2)
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            dataset.append(tmp)


        # st.write(dataset)
        # st.write(dataset)

        def dist(a, b):
            return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


        # dist_min
        def dist_min(Ci, Cj):
            return min(dist(i, j) for i in Ci for j in Cj)


        # dist_max

        def dist_max(Ci, Cj):
            return max(dist(i, j) for i in Ci for j in Cj)


        # dist_avg

        def dist_avg(Ci, Cj):
            return sum(dist(i, j) for i in Ci for j in Cj) / (len(Ci) * len(Cj))


        def find_Min(M):
            min = 1000
            x = 0
            y = 0
            for i in range(len(M)):
                for j in range(len(M[i])):
                    if i != j and M[i][j] < min:
                        min = M[i][j]
                        x = i
                        y = j
            return (x, y, min)


        def AGNES(dataset, dist, k):
            C = []
            M = []
            for i in dataset:
                Ci = []
                Ci.append(i)
                C.append(Ci)
            # print(C)
            # st.write(C)
            for i in C:
                Mi = []
                for j in C:
                    #             print(Mi)
                    Mi.append(dist(i, j))
                M.append(Mi)
            #     print(len(M))
            q = len(dataset)
            #     print(q)
            while q > k:
                x, y, min = find_Min(M)
                #         print(find_Min(M))
                # C is the cluster
                # M Distance matrix
                C[x].extend(C[y])
                C.remove(C[y])
                M = []
                for i in C:
                    Mi = []
                    for j in C:
                        Mi.append(dist(i, j))
                    M.append(Mi)
                q -= 1
            st.write(M)
            return C


        def draw(C):
            st.subheader("Plot of cluster using AGNES")
            colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
            c = ["Setosa", "Versicolor", "Virginica"]
            for i in range(len(C)):
                coo_X = []
                coo_Y = []
                for j in range(len(C[i])):
                    coo_X.append(C[i][j][0])
                    coo_Y.append(C[i][j][1])
                pl.xlabel(attribute1)
                pl.ylabel(attribute2)
                # pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i)
                pl.scatter(
                    coo_X, coo_Y, color=colValue[i % len(colValue)], label=i)

            pl.legend(loc='upper right')
            st.pyplot()


        n = st.number_input('Insert value for K', step=1, min_value=1)
        # st.write('The current number is ', number)
        C = AGNES(dataset, dist_avg, n)
        draw(C)
        st.subheader("Dendogram plot")
        dist_sin = linkage(iris.data, method="ward")
        plt.figure(figsize=(20, 15))
        dendrogram(dist_sin, above_threshold_color='#070dde',
                   orientation='right', leaf_rotation=90)
        plt.xlabel('Distance')
        plt.ylabel('Index')
        plt.title("Dendrogram", fontsize=18)
        plt.show()
        st.pyplot()

        # C = np.array(C,dtype=np.double)
        # st.write(C)
        # Z = hierarchy.linkage(C, method='average')
        # plt.figure()
        # plt.title("Dendrograms")
        # dendrogram = hierarchy.dendrogram(Z)
    if operation == 'DIANA':
        arr = []
        n = 0

        cols = []
        for i in data.columns[:-1]:
            cols.append(i)
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols, index=2)
        for i in range(len(data)):
            arr.append([data.loc[i, attribute1], data.loc[i, attribute2]])
        n = len(arr)
        k = int(st.number_input("Enter no of Clusters (k): ", min_value=1, step=1))

        # arr = []
        # n=0
        # for i in X:
        #   arr.append([i[0],i[1]])
        #   n += 1

        # print(X)
        # print("------------")
        # print(arr[0], arr[1])
        # print("------------")
        # print(atr2)

        # arr = np.array([[1, 2],[3,2],[2,5],[1,3],[6,5],[7,5],[4,6],[3,5],[4,1],[5,6],[3,8],[8,5]])
        # k = 3
        minPoints = 0
        if len(arr) % k == 0:
            minPoints = len(arr) // k
        else:
            minPoints = (len(arr) // k) + 1
        # print(len(arr))
        print(minPoints)


        def Euclid(a, b):
            # print(a,b)
            # finding sum of squares
            sum_sq = np.sum(np.square(a - b))
            return np.sqrt(sum_sq)


        points = [[0]]


        def findPoints(point):
            max = 0
            pt = -1
            for i in point:
                for j in range(len(arr)):
                    if j in point:
                        continue
                    else:
                        # print(arr[i], arr[j])
                        dis = Euclid(np.array(arr[i]), np.array(arr[j]))
                        if max < dis:
                            max = dis
                            # print(max)
                            pt = j
            return pt


        travetsedPoints = [0]
        for i in range(0, k):
            if len(travetsedPoints) >= len(arr):
                break

            # if len(points)>=k:
            #   break

            while (len(points[i]) < minPoints):
                # while(True):
                pt = findPoints(travetsedPoints)
                if pt in travetsedPoints:
                    break
                travetsedPoints.append(pt)
                points[i].append(pt)
            points.append([])
        points.remove([])
        # st.write(points)

        # colarr = ['blue','green','red','black']

        colarr = []

        for i in range(k):
            colarr.append('#%06X' % randint(0, 0xFFFFFF))

        i = 0
        cluster = []
        for j in range(k):
            cluster.append(j)

        # st.subheader("Cluster and Points")
        # # annotations=["Point-1","Point-2","Point-3","Point-4","Point-5"]
        # fig, axes = plt.subplots(1, figsize=(15, 20))
        # for atr in points:
        #     for j in range(minPoints):
        #         if atr[j]==-1:
        #             continue
        #     pltY = atr[j]
        #     pltX = cluster[i%(k+1)]
        #     # pltX = arr[atr[j]][0]
        #     # pltY = arr[atr[j]][1]
        #     # pltY = data.loc[:,classatr]
        #     plt.scatter(pltX, pltY, color=colarr[i])
        #     label = str("(" + str(arr[atr[j]][0]) + "," + str(arr[atr[j]][1]) + ")")
        #     plt.text(pltX, pltY, label)

        #     i += 1

        # plt.legend(loc=1, prop={'size':4})
        # # plt.show()
        # st.pyplot()

        j = 0


        def findIndex(ptarr):
            # print("Ptarr: ", ptarr)
            for j in range(len(points)):
                if ptarr in points[j]:
                    return j


        fig, axes = plt.subplots(1, figsize=(10, 7))
        clusters = []
        for i in range(k):
            clusters.append([[], []])

        for i in range(len(arr)):
            j = findIndex(i)
            clusters[j % k][0].append(arr[i][0])
            clusters[j % k][1].append(arr[i][1])

            # print(i)
            # plt.scatter(arr[i][0],arr[i][1], color = colarr[j])
        for i in range(len(clusters)):
            plt.scatter(clusters[i][0], clusters[i][1],
                        color=colarr[i % k], label=cluster[i])
        plt.title("Cluster plot using DIANA")
        plt.xlabel(attribute2)
        plt.ylabel(attribute1)
        plt.legend(loc=1, prop={'size': 15})

        # plt.legend(["x*2" , "x*3"])
        # plt.show()
        # st.subheader("Clustering using DIANA")
        st.pyplot()

        # st.write("Dendogram")
        # dismatrix =[]
        # for i in range(len(arr)):
        #     for j in range(i+1, len(arr)):
        #         dismatrix.append([Euclid(np.array(arr[i]),np.array(arr[j]))])
        # print(arr[i], arr[j])
        # print(arr[j])
        # ytdist = dismatrix
        st.subheader("Dendogram plot")
        dist_sin = linkage(iris.data, method="ward")
        plt.figure(figsize=(20, 15))
        dendrogram(dist_sin, above_threshold_color='#070dde',
                   orientation='left', leaf_rotation=90)
        plt.xlabel('Distance')
        plt.ylabel('Index')
        plt.title("Dendrogram", fontsize=18)
        plt.show()
        st.pyplot()

    if operation == "DBSCAN":

        def calDist(X1, X2):
            sum = 0
            for x1, x2 in zip(X1, X2):
                sum += (x1 - x2) ** 2
            return sum ** 0.5
            return (((X1[0] - X2[0]) ** 2) + (X1[1] - X2[1]) ** 2) ** 0.5


        def getNeibor(data, dataSet, e):
            res = []
            for i in range(len(dataSet)):
                if calDist(data, dataSet[i]) < e:
                    res.append(i)
            return res


        def DBSCAN(dataSet, e, minPts):
            coreObjs = {}
            C = {}
            n = dataset
            for i in range(len(dataSet)):
                neibor = getNeibor(dataSet[i], dataSet, e)
                if len(neibor) >= minPts:
                    coreObjs[i] = neibor
            oldCoreObjs = coreObjs.copy()
            # st.write(oldCoreObjs)
            # CoreObjs set of COres points
            k = 0
            notAccess = list(range(len(dataset)))

            # his will check the relation of core point with each other
            while len(coreObjs) > 0:
                OldNotAccess = []
                OldNotAccess.extend(notAccess)
                cores = coreObjs.keys()
                randNum = random.randint(0, len(cores))
                cores = list(cores)
                core = cores[randNum]
                queue = []
                queue.append(core)
                notAccess.remove(core)
                while len(queue) > 0:
                    q = queue[0]
                    del queue[0]
                    if q in oldCoreObjs.keys():
                        delte = [val for val in oldCoreObjs[q]
                                 if val in notAccess]
                        queue.extend(delte)
                        notAccess = [
                            val for val in notAccess if val not in delte]
                k += 1
                C[k] = [val for val in OldNotAccess if val not in notAccess]
                for x in C[k]:
                    if x in coreObjs.keys():
                        del coreObjs[x]
            st.write(C)
            return C


        def draw(C, dataSet):
            color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
            vis = set()
            for i in C.keys():
                X = []
                Y = []
                datas = C[i]
                for k in datas:
                    vis.add(k)
                for j in range(len(datas)):
                    X.append(dataSet[datas[j]][0])
                    Y.append(dataSet[datas[j]][1])
                plt.scatter(X, Y, marker='o',
                            color=color[i % len(color)], label=i)
            vis = list(vis)
            unvis1 = []
            unvis2 = []
            for i in range(len(dataSet)):
                if i not in vis:
                    unvis1.append(dataSet[i][0])
                    unvis2.append(dataSet[i][1])
            st.subheader("Plot of cluster's after DBSCAN ")
            plt.xlabel(attribute1)
            plt.ylabel(attribute2)
            plt.scatter(unvis1, unvis2, marker='o', color='black')
            plt.legend(loc='lower right')
            plt.show()
            st.pyplot()


        cols = []
        for i in data.columns[:-1]:
            cols.append(i)
        # atr1, atr2 = st.columns(2)
        attribute1 = st.selectbox("Select Attribute 1", cols)
        attribute2 = st.selectbox("Select Attribute 2", cols)
        dataset = []
        arr1 = []
        arr2 = []
        for i in range(len(data)):
            arr1.append(data.loc[i, attribute1])
        for i in range(len(data)):
            arr2.append(data.loc[i, attribute2])
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            dataset.append(tmp)
        r = st.number_input('Insert value for eps', value=0.09)
        mnp = st.number_input(
            'Insert mimimum number of points in cluster', step=1, value=7)
        C = DBSCAN(dataset, r, mnp)
        draw(C, dataset)

    if operation == "K-MEANS":
        cols = []
        for i in data.columns[:-1]:
            cols.append(i)
        atr1, atr2 = st.columns(2)
        attribute1 = atr1.selectbox("Select Attribute 1", cols)
        attribute2 = atr2.selectbox("Select Attribute 2", cols, index=1)


        # print(attribute1)
        # print(attribute2)

        class color:
            PURPLE = '\033[95m'
            CYAN = '\033[96m'
            DARKCYAN = '\033[36m'
            BLUE = '\033[94m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            BOLD = '\033[1m'
            UNDERLNE = '\033[4m'
            END = '\033[0m'


        def plot_data(X):
            plt.figure(figsize=(7.5, 6))
            for i in range(len(X)):
                plt.scatter(X[i][0], X[i][1], color='k')


        def random_centroid(X, k):
            random_idx = [np.random.randint(len(X)) for i in range(k)]
            centroids = []
            for i in random_idx:
                centroids.append(X[i])
            return centroids


        def assign_cluster(X, ini_centroids, k):
            cluster = []
            for i in range(len(X)):
                euc_dist = []
                for j in range(k):
                    euc_dist.append(np.linalg.norm(
                        np.subtract(X[i], ini_centroids[j])))
                idx = np.argmin(euc_dist)
                cluster.append(idx)
            return np.asarray(cluster)


        def compute_centroid(X, clusters, k):
            centroid = []
            for i in range(k):
                temp_arr = []
                for j in range(len(X)):
                    if clusters[j] == i:
                        temp_arr.append(X[j])
                centroid.append(np.mean(temp_arr, axis=0))
            return np.asarray(centroid)


        def difference(prev, nxt):
            diff = 0
            for i in range(len(prev)):
                diff += np.linalg.norm(prev[i] - nxt[i])
            return diff


        def show_clusters(X, clusters, centroids, ini_centroids, mark_centroid=True, show_ini_centroid=True,
                          show_plots=True):
            cols = {0: 'r', 1: 'b', 2: 'g', 3: 'coral', 4: 'c', 5: 'lime'}
            fig, ax = plt.subplots(figsize=(7.5, 6))
            for i in range(len(clusters)):
                ax.scatter(X[i][0], X[i][1], color=cols[clusters[i]])
            for j in range(len(centroids)):
                ax.scatter(centroids[j][0], centroids[j]
                [1], marker='*', color=cols[j])
                if show_ini_centroid == True:
                    ax.scatter(
                        ini_centroids[j][0], ini_centroids[j][1], marker="+", s=150, color=cols[j])
            if mark_centroid == True:
                for i in range(len(centroids)):
                    ax.add_artist(plt.Circle(
                        (centroids[i][0], centroids[i][1]), 0.4, linewidth=2, fill=False))
                    if show_ini_centroid == True:
                        ax.add_artist(plt.Circle(
                            (ini_centroids[i][0], ini_centroids[i][1]), 0.4, linewidth=2, color='y', fill=False))
            ax.set_xlabel(attribute1)
            ax.set_ylabel(attribute2)
            ax.set_title("K-means Clustering")
            if show_plots == True:
                plt.show()
                st.pyplot()
            # if show_plots==True:
            # plt.show()
            # st.pyplot()


        def k_means(X, k, show_type='all', show_plots=True):
            c_prev = random_centroid(X, k)
            cluster = assign_cluster(X, c_prev, k)
            diff = 10
            ini_centroid = c_prev

            # if show_plots:
            #     st.write("Initial Plot:")
            #     show_clusters(X, cluster, c_prev, ini_centroid,
            #                   show_plots=show_plots)
            while diff > 0.0001:
                cluster = assign_cluster(X, c_prev, k)
                if show_type == 'all' and show_plots:
                    show_clusters(X, cluster, c_prev, ini_centroid,
                                  False, False, show_plots=show_plots)
                    mark_centroid = False
                    show_ini_centroid = False
                c_new = compute_centroid(X, cluster, k)
                diff = difference(c_prev, c_new)
                c_prev = c_new
            st.write(
                "NOTE:\n +  Yellow Circle -> Initial Centroid\n * Black Circle -> Final Centroid")
            if show_plots:
                st.write("Initial Cluster Centers:")
                st.write(ini_centroid)
                st.write("Final Cluster Centers:")
                st.write(c_prev)
                st.write("Final Plot:")
                show_clusters(X, cluster, c_prev, ini_centroid,
                              mark_centroid=True, show_ini_centroid=True)
            return cluster, c_prev


        def validate(original_clus, my_clus, k):
            ori_grp = []
            my_grp = []
            for i in range(k):
                temp = []
                temp1 = []
                for j in range(len(my_clus)):
                    if my_clus[j] == i:
                        temp.append(j)
                    if original_clus[j] == i:
                        temp1.append(j)
                my_grp.append(temp)
                ori_grp.append(temp1)
            same_bool = True
            for f in range(len(ori_grp)):
                if my_grp[f] not in ori_grp:
                    st.write("Not Same")
                    same_bool = False
                    break
            if same_bool:
                st.write("Both the clusters are equal")


        k = st.number_input("Enter value for K", step=1, value=1)
        X, original_clus = make_blobs(
            n_samples=50, centers=3, n_features=2, random_state=len(attribute1))
        datat = []
        arr1 = []
        arr2 = []
        for i in range(len(data)):
            arr1.append(data.loc[i, attribute1])
        for i in range(len(data)):
            arr2.append(data.loc[i, attribute2])
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            datat.append(tmp)
        cluster, centroid = k_means(datat, k, show_type='ini_fin')

    if operation == "K-MEDOIDE":
        cols = []
        for i in data.columns[:-1]:
            cols.append(i)
        # atr1, atr2 = st.columns(2)
        attribute1 = st.selectbox("Select Attribute 1", cols)
        attribute2 = st.selectbox("Select Attribute 2", cols, index=1)


        class KMedoidsClass:
            def __init__(self, data, k, iters):
                self.data = data
                self.k = k
                self.iters = iters
                self.medoids = np.array([data[i] for i in range(self.k)])
                self.colors = np.array(np.random.randint(
                    0, 255, size=(self.k, 4))) / 255
                self.colors[:, 3] = 1

            def manhattan(self, p1, p2):
                return np.abs((p1[0] - p2[0])) + np.abs((p1[1] - p2[1]))

            def get_costs(self, medoids, data):
                tmp_clusters = {i: [] for i in range(len(medoids))}
                cst = 0
                for d in data:
                    dst = np.array([self.manhattan(d, md) for md in medoids])
                    c = dst.argmin()
                    tmp_clusters[c].append(d)
                    cst += dst.min()

                tmp_clusters = {k: np.array(v)
                                for k, v in tmp_clusters.items()}
                return tmp_clusters, cst

            def fit(self):

                self.datanp = np.asarray(data)
                samples, _ = self.datanp.shape

                self.clusters, cost = self.get_costs(
                    data=self.data, medoids=self.medoids)
                count = 0

                colors = np.array(np.random.randint(
                    0, 255, size=(self.k, 4))) / 255
                colors[:, 3] = 1

                plt.xlabel(attribute1)
                plt.ylabel(attribute2)
                [plt.scatter(self.clusters[t][:, 0], self.clusters[t][:, 1], marker="*", s=100,
                             color=colors[t]) for t in range(self.k)]
                plt.scatter(self.medoids[:, 0],
                            self.medoids[:, 1], s=200, color=colors)
                # plt.show()
                st.pyplot()

                while True:
                    swap = False
                    for i in range(samples):
                        if not i in self.medoids:
                            for j in range(self.k):
                                tmp_meds = self.medoids.copy()
                                tmp_meds[j] = i
                                clusters_, cost_ = self.get_costs(
                                    data=self.data, medoids=tmp_meds)

                                if cost_ < cost:
                                    self.medoids = tmp_meds
                                    cost = cost_
                                    swap = True
                                    self.clusters = clusters_
                                    st.write(
                                        f"Medoids Changed to: {self.medoids}.")
                                    st.subheader(f"Step :{count + 1}")
                                    # count += 1
                                    plt.xlabel(attribute1)
                                    plt.ylabel(attribute2)
                                    [plt.scatter(self.clusters[t][:, 0], self.clusters[t][:, 1], marker="*", s=100,
                                                 color=colors[t]) for t in range(self.k)]
                                    plt.scatter(
                                        self.medoids[:, 0], self.medoids[:, 1], s=200, color=colors)
                                    # plt.show()
                                    st.pyplot()
                    count += 1

                    if count >= self.iters:
                        st.write("End of the iterations.")
                        break
                    if not swap:
                        st.write("End.")
                        break


        # dt = np.random.randint(0,100, (100,2))
        datat = []
        arr1 = []
        arr2 = []
        for i in range(len(data)):
            arr1.append(data.loc[i, attribute1])
        for i in range(len(data)):
            arr2.append(data.loc[i, attribute2])
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            datat.append(tmp)
        k = st.number_input("Enter value fot k", step=1, min_value=1)
        kmedoid = KMedoidsClass(datat, k, 10)
        kmedoid.fit()


elif choice == 'Assignment No. 7':
    st.write("")
    st.header("Assignment No. 7")
    st.write("2019BTECS00067 - Vaibhav Kute")
    url = 'https://raw.githubusercontent.com/Udayraj2806/dataset/main/house-votes-84.data.csv'
    df = pd.read_csv(url)

    # st.write(df[:5])
    d = pd.DataFrame(df)
    data = d
    d.head()

    df_rows = d.to_numpy().tolist()

    cols = []
    for i in data.columns:
        cols.append(i)
    st.write(cols)
    col_len = len(cols)
    st.write("At Max Rules to be Generated: ",
             ((3**col_len)-(2**(col_len+1)))+1)
    st.write("Attributes:", len(cols))
    # st.write(cols)
    newDataSet = []
    # st.write(len(df_rows))
    i, cnt = 0, 0
    for row in df_rows:
        i += 1
        if '?' in row:
            continue
        else:
            lst = []
            cnt += 1
            for k in range(1, len(row)):
                if row[k] == 'y':
                    lst.append(cols[k])
            newDataSet.append(lst)
    st.write(newDataSet)

    # st.write(row)
    # st.write("--------------")
    # st.write(cnt)
    # st.write(newDataSet)
    # newDataSet.drop()

    data = []

    for i in range(len(newDataSet)):
        # data[i] = newDataSet[i]
        data.append([i, newDataSet[i]])

    st.write(data)

    # extract distinct items

    init = []
    for i in data:
        for q in i[1]:
            if (q not in init):
                init.append(q)
    init = sorted(init)

    st.write("Init:", len(init))

    # st.write(init)

    sp = 0.4
    s = int(sp*len(init))
    s

    from collections import Counter
    c = Counter()
    for i in init:
        for d in data:
            if (i in d[1]):
                c[i] += 1
    # st.write("C1:")
    for i in c:
        pass
        # st.write(str([i])+": "+str(c[i]))
    # st.write()
    l = Counter()
    for i in c:
        if (c[i] >= s):
            l[frozenset([i])] += c[i]
    # st.write("L1:")
    for i in l:
        pass
        # st.write(str(list(i))+": "+str(l[i]))
    # st.write()
    pl = l
    pos = 1
    for count in range(2, 1000):
        nc = set()
        temp = list(l)
        for i in range(0, len(temp)):
            for j in range(i+1, len(temp)):
                t = temp[i].union(temp[j])
                if (len(t) == count):
                    nc.add(temp[i].union(temp[j]))
        nc = list(nc)
        c = Counter()
        for i in nc:
            c[i] = 0
            for q in data:
                temp = set(q[1])
                if (i.issubset(temp)):
                    c[i] += 1
        # st.write("C"+str(count)+":")
        for i in c:
            pass
            # st.write(str(list(i))+": "+str(c[i]))
        # st.write()
        l = Counter()
        for i in c:
            if (c[i] >= s):
                l[i] += c[i]
        # st.write("L"+str(count)+":")
        for i in l:
            pass
            # st.write(str(list(i))+": "+str(l[i]))
        # st.write()
        if (len(l) == 0):
            break
        pl = l
        pos = count
    st.write("Result: ")
    st.write("L"+str(pos)+":")

    for i in pl:
        st.write(str(list(i))+": "+str(pl[i]))

    st.subheader("Rules Generation")
    for l in pl:
        st.write(l)
        break
    from itertools import combinations
    for l in pl:
        cnt = 0
        c = [frozenset(q) for q in combinations(l, len(l)-1)]
        mmax = 0
        for a in c:
            b = l-a
            ab = l
            sab = 0
            sa = 0
            sb = 0
            for q in data:
                temp = set(q[1])
                if (a.issubset(temp)):
                    sa += 1
                if (b.issubset(temp)):
                    sb += 1
                if (ab.issubset(temp)):
                    sab += 1
            temp = sab/sa*100
            if (temp > mmax):
                mmax = temp
            temp = sab/sb*100
            if (temp > mmax):
                mmax = temp
            cnt += 1
            st.write(str(cnt) + str(list(a))+" -> " +
                     str(list(b))+" = "+str(sab/sa*100)+"%")
            cnt += 1
            st.write(str(cnt) + str(list(b))+" -> " +
                     str(list(a))+" = "+str(sab/sb*100)+"%")
        mmax = st.number_input('Select value of confidence', step=5, min_value=5)
        mmax = int(mmax)
        curr = 1
        st.write("choosing:", end=' ')

        for a in c:
            b = l-a
            ab = l
            sab = 0
            sa = 0
            sb = 0
            for q in data:
                temp = set(q[1])
                if (a.issubset(temp)):
                    sa += 1
                if (b.issubset(temp)):
                    sb += 1
                if (ab.issubset(temp)):
                    sab += 1
            temp = sab/sa*100
            if (temp >= mmax):
                st.write(curr, end=' ')
            curr += 1
            temp = sab/sb*100
            if (temp >= mmax):
                st.write(curr, end=' ')
            curr += 1
            # break
        st.write()
        st.write()
        break



elif choice == 'Assignment No. 8':
    st.write("")
    st.header("Assignment No. 8")
    st.write("2019BTECS00067 - Vaibhav Kute")

    operation = st.radio("Select Operation", ["Select option","Web Crawler",'Page Rank','HITS algo'])
    if operation=="Select option":
        st.write("select the operation")
    elif operation=="Web Crawler":
        # def app(data):
        #     urls = 'https://www.geeksforgeeks.org/'
        title = st.text_input('Enter the link', 'Enter link here')
        # urls=input('Please enter the url: ')
        grab = requests.get(title)
        soup = BeautifulSoup(grab.text, 'html.parser')

        # opening a file in write mode
        # f = open("test1.txt", "w")
        # traverse paragraphs from soup
        for link in soup.find_all("a"):
            data = link.get('href')
            st.write(data)
    def printf(url):
        st.markdown(f'<p style="color:#000;font:lucida;font-size:20px;">{url}</p>', unsafe_allow_html=True)
    if operation == 'Page Rank':
        file = st.file_uploader("Upload dataset here", type=['csv','txt'], accept_multiple_files=False, disabled=False)
        dataset = pd.read_csv(file)

        st.dataframe(dataset.head(1000), width=1000, height=500)


        # Adjacency Matrix representation in Python

        class Graph(object):

            # Initialize the matrix
            def __init__(self, size):
                self.adjMatrix = []
                self.inbound = dict()
                self.outbound = dict()
                self.pagerank = dict()
                self.vertex = set()
                self.cnt = 0
                # for i in range(size+1):
                #     self.adjMatrix.append([0 for i in range(size+1)])
                self.size = size

            # Add edges
            def add_edge(self, v1, v2):
                if v1 == v2:
                    printf("Same vertex %d and %d" % (v1, v2))
                # self.adjMatrix[v1][v2] = 1
                self.vertex.add(v1)
                self.vertex.add(v2)
                if self.inbound.get(v2, -1) == -1:
                    self.inbound[v2] = [v1]
                else:
                    self.inbound[v2].append(v1)
                if self.outbound.get(v1, -1) == -1:
                    self.outbound[v1] = [v2]
                else:
                    self.outbound[v1].append(v2)

                # self.adjMatrix[v2][v1] = 1

            # Remove edges
            # def remove_edge(self, v1, v2):
            #     if self.adjMatrix[v1][v2] == 0:
            #         print("No edge between %d and %d" % (v1, v2))
            #         return
            #     self.adjMatrix[v1][v2] = 0
            #     self.adjMatrix[v2][v1] = 0

            def __len__(self):
                return self.size

            # Print the matrix
            def print_matrix(self):
                # if self.size < 1000:
                #     for row in self.adjMatrix:
                #         for val in row:
                #             printf('{:4}'.format(val), end="")
                #         printf("\n")
                #     printf("Inbound:")
                #     st.write(self.inbound)

                #     printf("Outbound:")
                #     st.write(self.outbound)
                # else:
                pass

            def pageRank(self):
                self.cnt = 0
                if len(self.pagerank) == 0:
                    for i in self.vertex:
                        self.pagerank[i] = 1 / self.size
                prevrank = self.pagerank
                # print(self.pagerank)
                for i in self.vertex:
                    pagesum = 0.0
                    inb = self.inbound.get(i, -1)
                    if inb == -1:
                        continue
                    for j in inb:
                        pagesum += (self.pagerank[j] / len(self.outbound[j]))
                    self.pagerank[i] = pagesum
                    if (prevrank[i] - self.pagerank[i]) <= 0.1:
                        self.cnt += 1

            def printRank(self):
                printf(self.pagerank)

            def arrangeRank(self):
                sorted_rank = dict(sorted(self.pagerank.items(), key=operator.itemgetter(1), reverse=True))
                # printf(sorted_rank)
                printf("PageRank Sorted : " + str(len(sorted_rank)))
                i = 1
                printf(f"Rank ___ Node ________ PageRank Score")
                for key, rank in sorted_rank.items():
                    if i == 11:
                        break
                    printf(f"{i} _____ {key} ________ {rank}")
                    i += 1

                # st.dataframe(sorted_rank)


        def main():
            g = Graph(7)
            input_list = []

            d = 0.5
            for i in range(len(dataset)):
                input_list.append([dataset.loc[i, 'fromNode'], dataset.loc[i, 'toNode']])
                g.add_edge(dataset.loc[i, 'fromNode'], dataset.loc[i, 'toNode'])
            size = len(g.vertex)
            if size <= 10000:
                adj_matrix = np.zeros([size + 1, size + 1])

                for i in input_list:
                    adj_matrix[i[0]][i[1]] = 1

                st.subheader("Adjecency Matrix")
                st.dataframe(adj_matrix, width=1000, height=500)

            printf("Total Node:" + str(len(g.vertex)))
            printf("Total Edges: " + str(len(input_list)))
            # for i in input_list:

            # g.print_matrix()

            i = 0
            while i < 5:
                if g.cnt == g.size:
                    break
                g.pageRank()
                i += 1
            # g.printRank()
            g.arrangeRank()


        main()

    if operation == 'HITS algo':
        file = st.file_uploader("Upload dataset here", type=['csv','txt'], accept_multiple_files=False, disabled=False)
        dataset = pd.read_csv(file)
        input_list = []

        st.subheader("Dataset")
        st.dataframe(dataset.head(1000), width=1000, height=500)
        vertex = set()
        for i in range(len(dataset)):
            input_list.append([dataset.loc[i, 'fromNode'], dataset.loc[i, 'toNode']])
            vertex.add(dataset.loc[i, 'fromNode'])
            vertex.add(dataset.loc[i, 'toNode'])
        size = len(vertex)
        adj_matrix = np.zeros([size + 1, size + 1])

        for i in input_list:
            adj_matrix[i[0]][i[1]] = 1

        printf("No of Nodes: " + str(size))
        printf("No of Edges: " + str(len(dataset)))
        st.subheader("Adjecency Matrix")
        st.dataframe(adj_matrix, width=1000, height=500)
        A = adj_matrix
        # st.dataframe(A)
        At = adj_matrix.transpose()
        st.subheader("Transpose of Adj matrix")
        st.dataframe(At)

        u = [1 for i in range(size + 1)]
        v = np.matrix([])
        for i in range(5):
            v = np.dot(At, u)
            u = np.dot(A, v)

        # u.sort(reverse=True)
        hubdict = dict()
        for i in range(len(u)):
            hubdict[i] = u[i]

        authdict = dict()
        for i in range(len(v)):
            authdict[i] = v[i]

        printf("Hub weight matrix (U)")
        st.dataframe(u)
        printf("Hub weight vector (V)")
        st.dataframe(v)
        hubdict = dict(sorted(hubdict.items(), key=operator.itemgetter(1), reverse=True))
        authdict = dict(sorted(authdict.items(), key=operator.itemgetter(1), reverse=True))
        # printf(sorted_rank)
        printf("HubPages : ")
        i = 1
        printf(f"Rank ___ Node ________ Hubs score")
        for key, rank in hubdict.items():
            if i == 11:
                break
            printf(f"{i} _____ {key} ________ {rank}")
            i += 1

        printf("Authoritative Pages : ")
        i = 1
        printf(f"Rank ___ Node ________ Auth score")
        for key, rank in authdict.items():
            if i == 11:
                break
            printf(f"{i} _____ {key} ________ {rank}")
            i += 1

        # u = sorted(u, reverse=True)
        # printf("Hub weight matrix (U)")
        # st.dataframe(u)
        # v = sorted(v, reverse=True)
        # printf("Hub weight vector Authority (V)")
        # st.dataframe(v[:11])

    else:
        st.write("Select the proper operation")




elif choice == 'Assignment No. 9 (mini project)':
    st.write("")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxxRTtMZsy0XX07nREDgnG6_hJb7NqdSyXHz7MwTb_wY0Wb06dz3d3t44GpWlMUc_B5c4&usqp=CAU");
    background-size: 100%;
    background-position: top left;
    background-repeat: true;
    background-attachment: local;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # variables
    disrad = False

    st.image(
        "https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTJ8fGNhZmV8ZW58MHx8MHx8&auto=format&fit=crop&w=500&q=60",
        width=800)

    st.header("Cafe related Trends")

    st.header(" Data Mining Mini project ")
    st.write("1. 2019BTECS00067 Vaibhav Kute")
    st.write("2. 2019BTECS00068 Aditi Joshi")
    st.write("3. 2019BTECS00072 Nikhil Khavanekar")
    st.write("4. 2019BTECS00073 Abhishek Kamble")

    file_meta_data = st.file_uploader("Upload transactions here", type=['csv'], accept_multiple_files=False,
                                      disabled=False)

    df = pd.read_csv(file_meta_data)
    st.subheader("Transaction related trends")
    # st.dataframe(df, width=2000, height=500)

    # file_transactions = st.file_uploader("Upload transactions here", type=['csv'], accept_multiple_files=False,disabled=False)
    #
    # df = pd.read_csv(file_transactions)
    # st.subheader("Dataset loaded Table")
    # st.dataframe(df, width=2000, height=500)
    #
    # file_holidays = st.file_uploader("Upload date info here", type=['csv'], accept_multiple_files=False,disabled=False)
    #
    # df = pd.read_csv(file_holidays)
    # st.subheader("Dataset loaded Table")
    # st.dataframe(df, width=2000, height=500)

    ######################################################################

    trends = st.radio('Select the trend :', (
    'Select the trend','Categories', 'Sell Catogory and Quantity sold', 'Catogory and Price', 'Qauntity sold and price',
    'Sell id and quantity', 'Sell id and price', 'Price elasticity'))
    # st.write('You selected:', Host_Country)

    # if trends == 'Categories':
    #
    #
    # elif trends == "Sell Catogory and Quantity sold":
    # elif trends == "Catogory and Price":
    # elif trends == "Qauntity sold and price":
    # elif trends == "Sell id and quantity":
    # elif trends == "Sell id and price":
    # elif trends == "Price elasticity":
    # elif trends == "Sell Catogory and Quantity sold":
    # elif trends == "Sell Catogory and Quantity sold":
    # elif trends == "Sell Catogory and Quantity sold":
    # else:
    # 	print("Please choose correct answer")
    if trends=='Select the trend':
        st.subheader("Select the trend to see")
    elif trends == 'Categories':
        st.subheader("Categories")
        count_test = df['SELL_CATEGORY'].value_counts()
        labels = df['SELL_CATEGORY'].value_counts().index
        fig = plt.figure(figsize=(6, 6))
        plt.pie(count_test, labels=labels, autopct='%1.1f%%')
        plt.legend(labels)
        plt.show()
        st.pyplot(fig)

    elif trends == "Sell Catogory and Quantity sold":
        # 1
        st.subheader("Sell Catogory and Quantity sold")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-3]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[-1]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Sell Catogory and Quantity sold")
        plt.show(block=True)
        st.pyplot()




    elif trends == "Catogory and Price":

        # 2
        st.subheader("Catogory and Price")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-4]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[-1]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Catogory and Price")
        plt.show(block=True)
        st.pyplot()

    elif trends == "Qauntity sold and price":

        # 3
        st.subheader("Qauntity sold and price")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-3]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[-4]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Qauntity sold and price")
        plt.show(block=True)
        st.pyplot()

    elif trends == "Sell id and quantity":

        # 4
        st.subheader("Sell id and quantity")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-3]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[-2]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Sell id and quantity")
        plt.show(block=True)
        st.pyplot()

    elif trends == "Sell id and price":

        # 5
        st.subheader("Sell id and price")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-4]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[-2]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Sell id and price")
        plt.show(block=True)
        st.pyplot()

    ###########################################################################
    elif trends == "Price elasticity":

        st.subheader("Price elasticity")

        data = df

        data.sort_values(by=['SELL_ID'], inplace=True)

        data["% Change in Demand"] = df["QUANTITY"].pct_change()
        data["% Change in Price"] = df["PRICE"].pct_change()
        data["Price Elasticity"] = data["% Change in Demand"] / data["% Change in Price"]
        # print(data)
        st.write(data)

    # print(data)
    else:
        st.header("Please choose correct answer")

    ###########################################################################

    file_holidays = st.file_uploader("Upload date info here", type=['csv'], accept_multiple_files=False, disabled=False)

    df = pd.read_csv(file_holidays)
    st.subheader("Date info related trend")
    # st.dataframe(df, width=2000, height=500)

    trends = st.selectbox('Select the trend related to date info:', (
    'Select the trend','HOLIDAYs', 'Year and Temperature', 'School Breaks', 'Is outdoor and temperature', 'Is Outdoor',
    'Holiday and average temperature', 'Is Weekends'))
    # st.write('You selected:', Host_Country)
    #
    # if trends == 'HOLIDAYs':
    # elif trends == "Year and Temperature":
    # elif trends == "School Breaks":
    # elif trends == "Is outdoor and temperature":
    # elif trends == "Is Outdoor":
    # elif trends == "Holiday and average temperature":
    # elif trends == "Is Weekends":
    # elif trends == "Sell Catogory and Quantity sold":
    # elif trends == "Sell Catogory and Quantity sold":
    # elif trends == "Sell Catogory and Quantity sold":
    # else:
    # 	print("Please choose correct answer")
    if trends=='Select the trend':
        st.subheader("Select the trend to see")
    elif trends == 'HOLIDAYs':
        st.subheader("HOLIDAYs")
        count_test = df['HOLIDAY'].value_counts()
        labels = df['HOLIDAY'].value_counts().index
        fig = plt.figure(figsize=(6, 6))
        plt.pie(count_test, labels=labels, autopct='%1.1f%%')
        plt.legend(labels)
        plt.show()
        st.pyplot(fig)



    elif trends == "Year and Temperature":

        # 1
        st.subheader("Year and Temperature")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-2]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[1]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Year and Temperature")
        plt.show(block=True)
        st.pyplot()


    elif trends == "School Breaks":

        st.subheader("School Breaks")
        count_test = df['IS_SCHOOLBREAK'].value_counts()
        labels = df['IS_SCHOOLBREAK'].value_counts().index
        fig = plt.figure(figsize=(6, 6))
        plt.pie(count_test, labels=labels, autopct='%1.1f%%')
        plt.legend(labels)
        plt.show()
        st.pyplot(fig)

    elif trends == "Is outdoor and temperature":

        # 2
        st.subheader("Is outdoor and temperature")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-2]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[-1]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Is outdoor and temperature")
        plt.show(block=True)
        st.pyplot()


    elif trends == "Is Outdoor":

        st.subheader("Outdoor")
        count_test = df['IS_OUTDOOR'].value_counts()
        labels = df['IS_OUTDOOR'].value_counts().index
        fig = plt.figure(figsize=(6, 6))
        plt.pie(count_test, labels=labels, autopct='%1.1f%%')
        plt.legend(labels)
        plt.show()
        st.pyplot(fig)

    elif trends == "Holiday and average temperature":

        # 3
        st.subheader("Holiday and average temperature")

        atr1, atr2 = st.columns(2)
        attribute1 = df.columns[-2]
        # atr1.selectbox("Select Attribute 1",df.columns[-2])
        # attribute2 = atr2.selectbox("Select Attribute 2", cols)
        # printf(attribute1)
        # printf(attribute2)
        classatr = df.columns[2]
        sns.set_style("whitegrid")
        sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
        plt.title("Holiday and average temperature")
        plt.show(block=True)
        st.pyplot()


    elif trends == "Is Weekends":

        st.subheader("Weekends")

        count_test = df['IS_WEEKEND'].value_counts()
        labels = df['IS_WEEKEND'].value_counts().index
        fig = plt.figure(figsize=(6, 6))
        plt.pie(count_test, labels=labels, autopct='%1.1f%%')
        plt.legend(labels)
        plt.show()
        st.pyplot(fig)
    else:
        st.write("Please choose correct answer")
    ######################################################


else :
    st.write("Select the proper option")

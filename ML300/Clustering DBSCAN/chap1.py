'''iris 데이터로 기본적인 군집분석 원리와 분석 과정을 연습해보자

군집분석이란? :
unsupervised learning 으로 데이터간의 유사도를 정의하고 그 유사도에 가까운 것부터 순서대로 합쳐 가는 방법이며, 유사도의 정의에는 거리나 상관계수등 여러 가지가 있다.

군집분석의 기본지식 clustering  1) centroid based 2)Density based

labeling이 되어있지 않은 데이터를 라벨링을 시켜보자
유사도를 정의하고 유사도를 기점으로 순서대로 합쳐가며 그룹핑을 해본다.

1.센트로이드 2.덴시티 방식
1 =  클러스터의 중심을 갖고 그 중심에서 데이터가 가까이 있고 없고의 차이를
2 = 얼마나 점들이 밀집되어있는 과정을 가지고 한다.

flat 이나 hierarchical 방식이 있는데
1 = flat에는 k-means가 유명하다 계층 방식을 가져가는 방법이 있다.

학습 목표
- 군집분석 의 다양한 알고리즘 경험
알고리즘 훈련에 필요한 파라미터를 알맞게 조절
알고리즘 훈련 결과를 시각화, 수치화를 통해 평가
서로 다른 알고리즘의 차이점을 이해해
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer
import sys

# iris 데이터셋 불러오기
# iris 데이터셋: 꽃받침과 꽃잎 사이즈로 3가지 종류의 case로 분류되어있는 데이터.
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df.head()

# 데이터를 dataframe화 시키기.
# array 형태를 Dataframe으로 변환.

'''문제 1. EDA - 컬럼명 재할당하기
# 방법1: 순서대로 입력.
# iris_df.columns = column_name_lst

# 방법2: dictionary형태를 사용하여 변경.

'''
# 컬럼명을 사용하기 편하게 변경해보세요
column_name_lst = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

column_replace_dict = {k:v for k, v in zip(iris.feature_names, column_name_lst)}
iris_df.rename(column_replace_dict, axis='columns',inplace=True)
iris_df.head()
# target 컬럼을 추가해보세요.
# target label은 iris.target에서 확인가능.
iris_df["target"] = iris.target

# target 종류를 확인해보세요.
iris_df.target.unique()

'''문제 2. EDA 결측값 확인하기
:각 컬럼의 결측값을 확인하고 결측값을 채우줍니다. 
'''
iris_df.isnull().sum()

#간단하게 컬럼별 결측값 유무 확인해보세요.

'''문제 3. EDA - 데이터 시각화 해보기
: 각 컬럼의 특징을 파악하고 시각화를 통해 데이터의 분포를 확인합니다.
'''
iris_df.dtypes
print(iris_df.describe())
import plotly.express as px


# scatter plot 생성.
fig = px.scatter(iris_df, x="sepal_width", y="sepal_length")
# 그래프 사이즈 조절.
fig.update_layout(width=600, height=500)
# 그래프 확인.
fig.show()

# scatter plot에 target컬럼 색으로 나타내기.
fig = px.scatter(iris_df, x="sepal_width", y="sepal_length", color="target")
fig.update_layout(width=600, height=500)
fig.show()

# 다른 컬럼으로 scatter plot 만들기.
fig = px.scatter(iris_df, x="petal_width", y="petal_length", color="target")
fig.update_layout(width=600, height=500)
fig.show()

'''Step2: Clustering : K-Means 알고리즘

문제 4. Clustering : K-Means - K-Means 모듈 탐색하기.
{TODO: 여기에 간략하게 k-means 원리를 설명할 예정, 앞강의에 k-means가 언급되므로 간략하게만}

'''
# K-Means 모듈을 import 합니다.
from sklearn.cluster import KMeans
# help(KMeans)


'''문제 5. Clustering : K-Means - train, test set으로 분리하기.'''
X = iris_df.iloc[:,:-1]
Y = iris_df.iloc[:,-1]

train_x, test_x ,train_y,test_y = train_test_split(X,Y, test_size=0.2)
print(len(train_x),len(test_x))


'''문제 6. clustering : K-means - K- means모듈 훈련시키기'''
# target이 3개인 것을 앞서 확인했지만, 5개의 그룹으로 clustering을 해봅니다.


km = KMeans(n_clusters = 5)
print(km)
# train set을 훈련시키고 cluster 결과를 확인합니다.
km.fit(train_x)
clusters_array = km.labels_

# 실제 iris데이터의 그룹과 훈련된 cluster의 결과를 비교해봅니다.
compare_cluster = dict(zip(clusters_array,train_y))
print(compare_cluster)
# 훈련된 label을 기준으로 시각화해보세요.
fig = px.scatter(x=train_x["petal_width"], y=train_x["petal_length"], color=clusters_array)
fig.update_layout(width=600, height=500)
fig.show()

fig = px.scatter(x=train_x["sepal_width"], y=train_x["sepal_length"], color=clusters_array)
fig.update_layout(width=600, height=500)
fig.show()

'''Sum of squared distances of samples to their closest cluster center.
[참고] k-means 알고리즘의 특성상, 훈련할때마다 km.inertia_ 값도 차이가 생깁니다. k값을 모르기 때문입니다.
각 군집마다 중심이 있고 중심의 포인트마다 거리를 sum을 하게된것을 k-mean에서 받아올수 있다. 
.'''

km.inertia_

'''문제 7. Clustering : K-Means - 최적의 k 찾기 (Elbow method).
: unsupervised 방법이므로 실제로 k(cluster 수)는 정해져 있지 않습니다. 따라서 최적의 k를 찾는 것부터 시작하게 됩니다.

최적의 k를 찾는 것이 k-means의 전체과정 중에서 가장 중요한 step입니다.
"Elbow method"란?
: **Total intra-cluster variation (or total within-cluster sum of square (=WSS))**가 최소가 되는 k를 찾는 방법.
'''
# elbow method를 사용하여 최적의 k를 찾아봅시다.
# k와 sum of squared distances from each point의 합(distortions)을 비교합니다.

distortions = []
k_range = range(1,11)
for i in k_range:
    km = KMeans(n_clusters=i)
    km.fit(train_x)
    distortions.append(km.inertia_)

print(distortions)

''' elbow method를 그래프를 통해 이해해 봅시다.
x축이 k의 수, y축이 distortions인 line plot을 그려봅시다. 기울기가 변화가 가장 작은것을 최적의 k라고 지정하는 방법

k가 클수록 군집으로 사용하기 좋지 않다 센터 즉 군집의 중심점으로부터 멀어지면 한 군집으로 들어가기 적합하지 않다고 한다. 
무조건 작으면 좋은것도 아닌것이 각 포인트들이 모든 개별의 클러스터가 되면 수치가 낮아 지기때문에 무조건 낮은것도 좋은것은 아니다.
클러스터가 한개일때 2개일때 3개 즉 기울기가 가장 변화가 적은것을 엘보우 메소드를 사용한다. 

수치가 많이 떨어지는것은 좋은것 (적합하다) 그래프가 명확할때는 눈으로 보기 좋지만 다른 모듈을 사용함으로써 비교해보자 
'''

fig = px.line(x=k_range, y=distortions, labels={"x":"k", "y":"distortions"})
fig.update_layout(width=800, height=500)
fig.show()


'''문제 8. Clustering : K-Means - 최적의 k 찾기 (KElbowVisualizer 사용해보기).
: model 훈련과 함께 그래프를 그려주고 훈련 시간까지 확인해주는 모듈인 KElbowVisualizer를 사용해봅니다.

Yellowbrick extends the Scikit-Learn API to make a model selection and hyperparameter tuning easier.'''

# KElbowVisualizer 사용해서 훈련(training)과 그래프를 한번에 해결해보세요.
# elbow mothod를 시각화도 시켜주고 수치적으로 판단해주는 모듈

km = KMeans()
visualizer = KElbowVisualizer(km, k=(1,11))
visualizer.fit(train_x)
visualizer.poof()
'''클러스트 갯수를 정하지 않은 k를 주고 '''
'''문제 9. Clustering : K-Means - 최적의 k 찾기 (kneed 모듈 사용해보기).
: 그래프를 확인하지 않고도 최적의 k값을 자동으로 찾아주는 모듈인 kneed를 사용해봅니다.'''
# 아래 parameter를 참고하여 kneed 모듈을 사용하여 자동으로 elbow값을 찾아보세요.

from kneed import KneeLocator
"""
[KneeLocator parameter 참고]
curve (str) – If ‘concave’, algorithm will detect knees. If ‘convex’, it will detect elbows.
direction (str) – one of {“increasing”, “decreasing”}
s 는 - 
curve는 - 어떤 형태의 굴곡이냐를 2가지 파라미터를 주고
direction 은 방향성 
output을 보려면 kneedle.elbow  x와 y 를 이전 그래프의 기준으로 보면 꼭지점을 알려주는 k
y를 확인하는것은 kneedle.elbow_y 를 주고 

# 볼록 curve plot의 경우, 아래와 같이 knee를 찾습니다.
# print(round(kneedle.knee, 3))
# print(round(kneedle.knee_y, 3))
"""

kneedle = KneeLocator(x=k_range, y=distortions, S=1.0, curve="convex", direction="decreasing")
print(f"최적의 k : {round(kneedle.elbow, 3)}")
print(f"최적의 k의 y값 : {round(kneedle.elbow_y, 3)}")
kneedle.plot_knee()



'''문제 10. K-Means - 최적의 k 찾기 (Silhouette method).
: 최적의 k를 찾는 다른 방법으로, cluster내의 거리가 아닌 cluster간의 거리도 함께 고려한 계수를 사용해서 
최적의 k를 비교해봅니다.

"Silhouette method"란?
: cluster내의 거리와 cluster간의 거리를 사용한 계수로 Silhouette coefficient(SC)값이 최대가 되는 k를 찾는 방법.

SC 해석?
: 각 cluster 사이의 거리가 멀고 cluster 내 데이터의 거리가 작을수록 군집 분석의 성능이 좋음.
 Silhouette 값은 -1에서 1까지 가능하며, 
0일 경우에는 cluster간의 변별력이 없다는 의미. -1에 가까울수록 clustering의 결과가 좋지 않음을 의미.


metric 는 euclidean을 많이 사용한다. 
'''


# silhouette_score 모듈을 사용해봅니다.
# [주의!] silhouette_score는 array 형태를 읽을 수 있습니다.
# [주의!] 군집간의 거리 계산을 필요로 하기때문에, 최소 2개이상의 label/cluster가 있어야 합니다.
from sklearn.metrics import silhouette_score

slihouette_score = []
# elbow 와 다르게 군집간의 거리를 계산하기대문에 k_range가 항상 1보다 커야한다 그래야 거리를 잴수있기때문입니다.
k_range = range(2,11)
for i in k_range:
    km = KMeans(n_clusters=i)
    km.fit(train_x)
    label = km.predict(train_x) #kmean훈련시키고 pred해서 트레이닝 y값을 다시 넣어주면 이것에 대한 label이 나오게된다. 긜고 수치를 저장해준다.
    sc_value = slihouette_score(np.array(train_x),label, metrix ="euclidean", sample_size=None, random_state=None)
    slihouette_score.append(sc_value)
print(slihouette_score)


# Silhouette method를 그래프를 통해 이해해 봅시다.
# x축이 k의 수, y축이 silhouette scores line plot을 그려봅시다.
fig = px.line(x=k_range, y=slihouette_score, labels={"x":"k", "y":"Silhouette scores"})
fig.update_layout(width=800, height=500)
fig.show()

'''문제 11. Clustering : K-Means - 최적의 k 찾기 (SilhouetteVisualizer 사용해보기).
: KElbowVisualizer와 유사한 SilhouetteVisualizer를 사용해보고 Elbow method와는 다른 그래프를 해석하는 방법을 배워봅니다.

silhouette score만 보는 것이 아닌, 그래프를 통해 각 군집의 분포를 종합적으로 평가합니다.

# SilhouetteVisualizer 사용해서 훈련(training)과 그래프를 한번에 해결해보세요.
여기서는 array 형태가 아니어도 됩니다. 
'''


from yellowbrick.cluster import SilhouetteVisualizer
k_range=range(2,6)
for i in k_range:
    km=KMeans(n_clusters=i)
    visualizer = SilhouetteVisualizer(km)
    visualizer.fit(train_x)
    visualizer.poof()


'''문제 12. Clustering : K-Means - 최적의 k를 사용하여 모델 훈련시키기.
: 위에서 구한 최적의 k를 사용하여 다시 모델을 훈련시켜봅니다.'''
# 위에서 찾은 최적을 k를 할당하고 k-means model 훈련을 시켜보세요.

k = 3

km = KMeans(n_clusters = k).fit(train_x)

train_cluster = km.labels_



'''문제 13. Clustering : K-Means - 훈련된 cluster를 그래프로 비교해보기.
: 훈련시킨 k-means model의 cluster 결과를 원래의 label과 비교해봅시다.'''

# 실제 label과 훈련된 결과 cluster를 그래프로 비교해보기.
# plotly에서 subplot을 만들기.
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2, subplot_titles=("Actual","K-means cluster"))
fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=train_y),
               ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=train_cluster),
               ),
    row=1, col=2
)

fig.update_layout(height=600, width=800)
fig.show()
'''2개의 그래프를 보면 색이 다릅니다. 군집이 나눠진것에서 조금의 차이가 있습니다. 대부분 비슷하지만
 
 다른 클러스터에 할당이 된것과 같은 클러스터의 할당된것들까지 있습니다
 
 밑에 것에서 표를 보면 대부분 비슷하지만 가운데 부분에서 조금 섞여있는것들 즉 다른것들을 보실수있습니다. 
 '''

fig = make_subplots(rows=1, cols=2, subplot_titles=("Actual","K-means cluster"))

fig.add_trace(
    go.Scatter(x=train_x["petal_width"],
               y=train_x["petal_length"],
               mode="markers",
               marker=dict(color=train_y),
               ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=train_x["petal_width"],
               y=train_x["petal_length"],
               mode="markers",
               marker=dict(color=train_cluster),
               ),
    row=1, col=2
)

fig.update_layout(height=600, width=800)
fig.show()
'''문제 14: Clustering : K-Means - 훈련된 모델에 test set을 사용해 predict 하기.
: 위의 model이 잘 훈련되었다는 가정하에, test set을 사용하여 모델을 평가해봅시다. '''

# [문제 12]의 모델을 그대로 사용하여 prediction을 해보세요 훈련과 함께 할수있는 fit_predict를 사용해준다. 바로 라벨이 나오게된다.
# 테스트셋에 대한 예측값들이 나오고 위에와 동일하게 비교를 해봅니다. 코드는 동일합니다. .
test_cluster = km.fit_predict(test_x)
print(test_cluster)
print(list(test_y))
# [문제 13]과 동일하게 그래프로 prediction결과를 비교해보세요.
fig = make_subplots(rows=1, cols=2, subplot_titles=("Actual-test","K-means cluster-test"))

fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=test_y),
               ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=test_cluster),
               ),
    row=1, col=2
)

fig.update_layout(height=600, width=800)


# [문제 13]과 동일하게 그래프로 prediction결과를 비교해보세요.
fig = make_subplots(rows=1, cols=2, subplot_titles=("Actual-test","K-means cluster-test"))

fig.add_trace(
    go.Scatter(x=train_x["petal_width"],
               y=train_x["petal_length"],
               mode="markers",
               marker=dict(color=test_y),
               ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=train_x["petal_width"],
               y=train_x["petal_length"],
               mode="markers",
               marker=dict(color=test_cluster),
               ),
    row=1, col=2
)

fig.update_layout(height=600, width=800)

'''그래프 많으로는 비교하기가 어려운것들이 있기때문에 결과를 수치적으로 평가합니다. 
문제 15. Clustering : K-Means - clustering 결과를 수치적으로 평가하기.
: 다차원일수록 그래프를 통해 clustering 결과를 확인하기 어렵기 때문에, 객관적인 수치로 평가하는 방법이 필요합니다.
'''
# test set의 accuracy score을 구해서 k-means 모델을 평가해보세요.
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_y,train_cluster)
test_acc= accuracy_score(test_y,test_cluster)
print(f"Accuracy score of train set : {round(train_acc, 4)}")
print(f"Accuracy score of test set : {round(test_acc, 4)}")

'''문제 16. Clustering : K-Means - 실제 cluster명과 매칭해서 accuracy 확인하기.

# 실제 cluster명과 매칭해주는 함수를 만들어보세요.
# [참고] scipy.stats.mode()를 사용합니다.


'''

import scipy

def find_matching_cluster(cluster_case,actual_labels,cluster_labels):
    matched_cluster={}
    actual_case = list(set(actual_labels)) #실제 라벨은 0,1,2 이기 때문에 리스트 형태로 실제 라벨 값을 가져오게됩니다. train_y가 됩니다.
    for i in cluster_case:
        idx = cluster_labels == i
        new_label = scipy.stats.mode()(actual_labels[idx])[0][0]
        actual_case.remove(new_label)
        matched_cluster[i] = new_label
        print(f"훈련된 라벨명은:{i} >> 가장 비번한 실제 label명:{new_label}")
    return matched_cluster
km_train_case = list(set(train_cluster))
print(km_train_case)
# 위의 함수를 사용해 train set의 cluster명을 다시 확인해보세요.
train_param_dict = find_matching_cluster(km_train_case, train_y,train_cluster)
print(train_param_dict)
# train과 유사하게 값이 나온다 어떤 라벨에 어떤값을 지정하는지 저장했기때문에
# 동일한 방법으로 test set의 cluster명을 다시 확인해보세요.
km_test_case = list(set(test_cluster))
test_perm_dict = find_matching_cluster(km_test_case, test_y, test_cluster)
print(list(test_y)[:10])
print(test_cluster[:10])
print(test_perm_dict)

train_new_labels=[train_param_dict[label]for label in train_cluster]
test_new_labels = [test_perm_dict[label]for label in test_cluster]


train_acc = accuracy_score(train_y, train_new_labels)
test_acc = accuracy_score(test_y, test_new_labels)
print(f"Accuracy score of train set : {round(train_acc, 4)}")
print(f"Accuracy score of test set : {round(test_acc, 4)}")

'''Step3: Clustering : Agglomerative 알고리즘 (계층군집)
: 데이터 자체의 분포와 어떻게 grouping을 하고자 하는지에 따라 다른 알고리즘이 사용될 수 있습니다.

Clustering : K-means VS Agglomerative 비교
[k-means 최적의 환경]
- 원형 혹은 구(spherical) 형태의 분포
- 동일한 데이터 분포 (size of cluster)
- 동일한 밀집도 (dense of cluster)
- 군집의 센터에 주로 밀집된 분포
- Noise와 outlier가 적은 분포
[k-means의 민감성]
- Noise와 outlier에 민감함.  
- 처음 시작하는 점에 따라 결과에 영향을 줌.  
- k값을 직접 설정해야하는 어려움이 있음.
'''\
    '''문제 17. Clustering : Agglomerative - Agglomerative 모듈 탐색&훈련시키기.
: k-means의 disadvantage를 보완할 수 있는 Agglomerative 알고리즘을 훈련시켜봅니다.

[AgglomerativeClustering 파라미터 참고사항]
- linkage 종류 : {‘ward’, ‘complete’, ‘average’, ‘single’}
- linkage="ward"이면, affinity="euclidean"만 가능.
- distance_threshold!=None 이면, n_clusters=None 이어야함.
- distance_threshold!=None 이면, compute_full_tree=True 이어야함.
'''
# Agglomerative 모듈 import
from sklearn.cluster import AgglomerativeClustering
# help(AgglomerativeClustering)
# 자주 사용되는 parameter를 사용하여, Agglomerative를 훈련시켜보세요.
# 파라미터 설정 시, 상단의 '참고사항' 부분을 확인해주세요.
# linkage="ward"
aggl = AgglomerativeClustering(n_clusters=3, linkage="ward", affinity="euclidean").fit(train_x)
print(aggl)

'''문제 18. Clustering : Agglomerative - 훈련된 cluster를 그래프로 비교해보기.'''
aggl_labels = aggl.labels_
# 원래의 label과 Agglomerative 알고리즘 결과를 시각화로 비교해보세요.
# 타이틀 앞에는 실제 그래프로 하고 뒤에꺼는 새로 훈련된 모델 클러스토로 해줍니다.
# 각각의 그래프를 추가를 해준다. 그리고 안에다가 실제 flat을 써주면됩니다.
fig = make_subplots(rows=1, cols=2, subplot_titles=("Actual","Agglomerative cluster"))
fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=train_y)),
    row=1,col=1
)
fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=aggl_labels)),
    row=1,col=2
)
fig.update_layout(height=600,width=800)
fig.show()
'''여기서 그래프를 보면 차이가 없어 보이지만 그 이유를 보면 지금 사용하고 있는 iris자체가 구형이기에 kmean가 잘 적용이 되는것이기에 
 이것도 완벽하게 똑같지는 않지만 조금씩 조금씩 다르고 색도 다르게 나타낸다. 
 
 다른 두 컬럼도 비교해보자  다 똑같고 변수명만 변경을 해주면된다. '''
fig.add_trace(
    go.Scatter(x=train_x["petal_width"],
               y=train_x["petal_length"],
               mode="markers",
               marker=dict(color=train_y)),
    row=1,col=1
)
fig.add_trace(
    go.Scatter(x=train_x["petal_width"],
               y=train_x["petal_length"],
               mode="markers",
               marker=dict(color=aggl_labels)),
    row=1,col=2
)
fig.update_layout(height=600,width=800)
fig.show()


'''
문제 19. Clustering : Agglomerative - clustering 결과를 수치적으로 평가하기.
실제 라벨을 가지고 있기에 실제 진행을 해줄수 있다. 


'''
# [문제 16]과 동일한 방법으로, cluster명을 매칭시켜보세요.
# label 종류를 저장하세요. 케이스만 가져오고 0,1,2 만 저장이 되어있습니다.

aggl_case = list(set(aggl_labels))
print(aggl_case)
# find_matching_cluster 함수를 사용해서 매칭되는 dictionary를 생성하세요.
aggl_perm_dict = find_matching_cluster(aggl_case, train_y, aggl_labels)
print(aggl_perm_dict)
# 생성한 dict 변수를 사용하여 훈련된 결과 label을 변경해주세요.
aggl_new_labels = [aggl_perm_dict[label] for label in aggl_labels]

# 트레인만 비교 해보기 train_y와 새로 만든 라벨의 유사도 비교
aggl_acc = accuracy_score(train_y,aggl_new_labels)
print(f"Accuracy score of K-means : {round(train_acc, 4)}")
print(f"Accuracy score of Agglomerative : {round(aggl_acc, 4)}")

'''문제 20. Clustering : Agglomerative - dendrogram을 그리기 위한 linkage matrix 구성 이해하기.'''

# dendrogram을 그리기 위한 matplotlib와 scipy의 dendrogram import.
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 샘플 데이터를 통해 linkage matrix 구조를 파악해봅시다.
sample_arr = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
print(sample_arr)
'''린케이지 메트릭스를 파악해보자. 

'''# 샘플 데이터로 linkage matrix를 생성해보세요.
sample_linkage = linkage(sample_arr, "single")
print(sample_linkage)
'''모듈중 dendrogram을 사용하여 linkage matrix를 사용해서 dengrogram을그려보겠습니다.

sample_linkage  데이터를 가지고 군집을 나누는 것을 표로 나타내는것을 이 20문제에서 표현을 했습니다. 
linkage 는 항상 4줄이 나오게됩니다. node1 node2 distance total nodoes 의 컬럼으로 표현된다. 
distance 는 y축과의 거리를 표현한것이다. 즉 5와6의 거리는 0이다 왜냐하면 x좌표에만 표시가 되어있기대문이기에.
토탈노드는 노드가 포함되어있는 공간안에 몇개의 노드가 들어있는지를 표현한것이다. 
11번재 노드라는것 안에 지표 수가 4개가 들어있다는것이다. 
마지막까지 이어나가게되면 다 이어지고 
군집을할때 가로선을 그으면 걸리는것이 2개이면 클러스터가 2개이고 만약에 y이 축에 1.5에 선을 그으면 나오는것은 3개가 나오게됩니다.
이런식으로 표현을 하게되고 dendrogram으로 표현을 합니다. 

'''

fig = plt.figure(figsize=(13,6))
dn= dendrogram(sample_linkage)

'''문제 21. Clustering : Agglomerative - dendrogram을 통해 알고리즘 이해하기.'''


# linkage_matrix를 생성하는 함수를 만들어보세요.
# [참고] model.children_, model.labels_, model.distances_ 를 활용하세요.
# [참고] dendrogram을 만들기위해 어떤 형태의 데이터가 필요한지 확인하세요.

def create_linkage(model):
  # 각 노드에 총 point수를 계산하기.
  counts = np.zeros(model.children_.shape[0]) # children 길이만큼 0 채운 array.모델이 가지고있는 칠드런 혹은 라벨
  n_samples = len(model.labels_) # 각 point의 cluster label.
  for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
      if child_idx < n_samples:
        current_count += 1  # leaf node
      else:
        current_count += counts[child_idx - n_samples]
    counts[i] = current_count

  linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
  return linkage_matrix
# Dendrogram을 그리기 위해서는 distance_threshold와 n_clusters 파라미터 조정이 필요합니다.
# distance_threshold=0, n_clusters=None 거리의 treshold를 두지 않아야 전체 그래프를 그릴수있습니다.
'''모든 케이스들을 다 받아내야 그릴수있기때문에 클러스트를 0으로 처리를 해줘야합니다. 그래프를 그리기 위해서는 함수를 지정해줘야합니다. 
그래서 위에서 함수도 만들어 놓은겁니다. '''
aggl_dend = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(train_x)


# create_linkage 함수를 사용해서 linkage matrix를 생성하고, dendrogram을 그려보세요.
# x축 - 실제 point(혹은 각 노드에 포함되는 point수)
plt.title('Agglomerative Clustering Dendrogram')
linkage_matrix = create_linkage(aggl_dend)
dendrogram(linkage_matrix, truncate_mode="level", p=3)
plt.show()
'''x축의 150개 200개 전체 데이터로 그려진게 아니고 x축의 숫자 즉 각노드들의 데이터 포인트 수입니다. '''
aggl = AgglomerativeClustering(n_clusters=3,linkage="ward",affinity="euclidean").fit(train_x)

aggl_labels=aggl.labels_
fig = make_subplots(rows=1, cols=2, subplot_titles=("Actual","Agglomerative cluster"))

fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=train_y),
               ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=train_x["sepal_width"],
               y=train_x["sepal_length"],
               mode="markers",
               marker=dict(color=aggl_labels),
               ),
    row=1, col=2
)

fig.update_layout(height=600, width=800, showlegend=False)
fig.show()

'''참고] Clustering : Agglomerative - 언제 사용해야하나요?
[Hierarchical clustering의 장단점]
- cluster수(k)를 정하지 않아도 사용가능.
- random point에서 시작하지 않으므로, 동일한 결과가 나옴.
- dendrogram을 통해 전체적인 군집을 확인할 수 있음 (nested clusters).

- 대용량 데이터에 비효율적임 (계산이 많음).

-> 샘플 데이터로 가볍게 군집분포를 확인하거나 nested clusters를 확인하기에 유용함.
-> 뒤에 배울 HDBSCAN의 기초지식.


Step4: Clustering : DBSCAN 알고리즘
Clustering : DBSCAN - DBSCAN 알고리즘 기초파악.
DBSCAN 알고리즘 (density-based spatial clustering of applications with noise) :

[DBSCAN 장점]
- K-means와 달리 최초 k(군집수)를 직접 할당하지 않음.
- Density(밀도)에 따라서 군집을 나누기 때문에, 기하학적인 모양을 갖는 분포도 적용 가능.
- Oulier 구분이 가능함.


문제 22. Clustering : DBSCAN - 비구형(nonspherical) 데이터 생성하기.

'''

# make_moons를 이용하여 DBSCAN 알고리즘을 적용시킬 비구형분포 데이터를 생성해보세요.
from sklearn.datasets import make_moons

# 샘플수와 noise정도 등의 파라미터를 입력할 수 있습니다.
moon_data, moon_labels = make_moons(n_samples=400, noise=0.1, random_state=42)
print(moon_data[:5])


# array 형태의 데이터를 Dataframe 형태로 변경해주세요.
moon_data_df = pd.DataFrame(moon_data, columns=["x", "y"])
moon_data_df["label"] = moon_labels
moon_data_df.head()
# scatter plot에 target컬럼 색으로 데이터를 나타내보세요.
fig = px.scatter(moon_data_df, x="x", y="y", color="label")
fig.update_layout(width=600, height=500)
fig.show()
'''위에서 노이즈를 0.9로 바꾸게되면 겹쳐져서 보이게 됩니다. 그래서 안겹쳐 보이게 0.1로 데이터를 하면 됩니다. 




문제 23. Clustering : DBSCAN - DBSCAN 알고리즘 탐색하기.
: 사용할 모듈을 import 하고, DBSCAN 알고리즘의 파라미터를 살펴봅니다.

[Parameters]

- eps(epsilon): 기준점부터의 반경. 
- min_samples: 반경내 있어야할 최소 data points.
입실로리는 데이터 한점에서 한 반지름의 거리이고 반지름을 동그라미 그렸을때 분포안에 민 샘플즈를 할당해주면 그 갯수만큼 들어가는것을
기준으로 잡고 군집을 생성하게 됩니다. 
한점의 기준에서 입실론 거리만큼  범위를 잡고 그 안에 min_sample을 갯수를 할당해주고 그 안에 개수를 뭐가 됫건 그안에 갯수가 들어가는것이 
기준이 되는것입니다.
그래서 모든 점을 3가지로 할당을 하게됩니다. 
- Core point
- Border point
- Outlier/Noise point

eps = 0.3 으로 주고 min_sample =6으로 주어졌을때 그 안에 샘플이 6개가 있으면 core point이고 eps를 옮기게 되어 sample이 6개가 되지 않을때는
border point 라고 합니다. 
그리고 outlier point 는 sample를 옮겻을때 그 범위 안에 자기 만 포함되어있다면 outlier point라고 하비낟. 

2가지 파라미터를 통해서 아웃라이어랑 클러스터를 얼마나 빡ㅃ가하게 만들수있는지 알려주는것이다. 

'''

# DBSCAN 모듈을 import 합니다.
from sklearn.cluster import DBSCAN
# help(DBSCAN)


# DBSCAN의 대표적인 파라미터인 eps(radius of neighborhood)와 min_samples(minimun number of data points)를 설정해봅시다.
# eps=0.2, min_samples=6
dbscan = DBSCAN(eps=0.2, min_samples=6)
# 위문제에서 생성한 moon_data를 훈련시켜보세요.
dbscan.fit(moon_data)
'''DBSCAN(algorithm='auto', eps=0.2, leaf_size=30, metric='euclidean',
       metric_params=None, min_samples=6, n_jobs=None, p=None)
       '''
dbscan_label = dbscan.labels_
dbscan_label[:10]
# array 형태이기때문에 set으로 한다 -1 부터 시작하는것은 outline을 잡은것으로 보면되기에 2개로
# 이렇게 훈련이 된것으로 파라미터를 변경해가면서 비교를 해보자 .
set(dbscan_label)
# 위의 결과를 plotly를 사용하여 시각화로 나타내보세요.
moon_data_df["dbscan_label"]=dbscan_label
fig = px.scatter(moon_data_df,x="x",y="y",color="dbscan_label")
fig.update_layout(width=600,height=500)
fig.show()

'''비교할때 함수를 하나 만들어서 사용을 해보자 
실제로는 파라미터 튜닝과정을 하는것이 많을거기 때문입니다.
밑에 for loop 과정에서 outlier을 빼고 계산 하고싶으면 -1의 여부를 추가해주면됩니다. 
'''
# eps 파라미터가 0.1, 0.2, 0.5일때 clusters의 차이를 비교해보세요.
for eps in [0.1,0.2,0.5]:
    dbscan = DBSCAN(eps=eps,min_samples=6).fit(moon_data)
    dbscan_label=dbscan.labels_
    print(f"epls:{eps}->>>label수: {len(set(dbscan_label))}")
    moon_data_df["dbscan_label"]=dbscan_label

    fig=px.scatter(moon_data_df,x='x',y='y',color="dbscan_label")
    fig.update_layout(width=600,height=500)
    fig.show()
# min_samples 파라미터가 2, 6, 30 일때 clusters의 차이를 비교해보세요.

for min_samples in [2, 6, 30]:
  dbscan = DBSCAN(eps=0.2, min_samples=min_samples).fit(moon_data)
  dbscan_label = dbscan.labels_
  print(f"min_samples:{min_samples} ->> label수: {len(set(dbscan_label))}")
  moon_data_df["dbscan_label"] = dbscan_label
  fig = px.scatter(moon_data_df, x="x", y="y", color="dbscan_label")
  fig.update_layout(width=600, height=500)
  fig.show()
'''
문제 26. Clustering : DBSCAN - DBSCAN와 K-means의 성능 비교하기.


'''
# 2개의 그룹으로 k-means를 훈련시켜보세요.
moon_km = KMeans(n_clusters = 2).fit(moon_data)
moon_km
'''KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=2, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
       '''
# [문제 6]와 같이 실제 label과 훈련된 label을 비교해보세요.
compare_kmeans_clusters = dict(zip(moon_km.labels_, moon_labels))
print(compare_kmeans_clusters)
# k-means 결과를 plotly를 사용하여 시각화로 나타내고, dbscan의 결과와 비교해보세요.
moon_data_df["kmeans_label"] = moon_km.labels_
moon_data_df["kmeans_label"] = moon_data_df["kmeans_label"].astype(str)

moon_dbscan = DBSCAN(eps=0.2, min_samples=6).fit(moon_data)
moon_data_df["dbscan_label"] = moon_dbscan.labels_
moon_data_df["dbscan_label"] = moon_data_df["dbscan_label"].astype(str)
for label_case in ["dbscan_label", "kmeans_label"]:
  fig = px.scatter(moon_data_df, x="x", y="y", color=label_case)
  fig.update_layout(width=600, height=500, title=f"{label_case} 시각화")
  fig.show()

'''문제 27. Clustering : DBSCAN - clustering 결과를 수치적으로 평가하기.'''
# 문제 15, 16과 같이, k-means 결과와 dbscan의 결과를 수치적으로 비교해보세요.
# 아래 수정된 "find_matching_cluster" 함수를 사용하여, DBSCAN 훈련결과 cluster명을 다시 확인해보세요.
# [참고] k를 정하지 않는 DBSCAN의 경우, label case 수가 다름을 감안해야합니다!
def find_matching_cluster(cluster_case,actual_labels,cluster_labels):
    matched_cluster={} #딕셔너리에 저장을 했습니다.
    temp_labels= [i+100 for i in range(100)] #list 형태의 템포라리형태의 라벨을 만들어줍니다.
    actual_case = list(set(actual_labels))#실제 케이스는 set으로 고유 값을 가져오는데 list형태로 저장해주비낟.    for i in cluster_case:
        #forloop 을 처음 시작하고 dbscan의 특징으로 라벨개수를 알지못하니깐 실제 할당받은 개수가 0보다 클경우 에는
        #아래에 짜논 함수가 돌아가고 그렇치 아는경우 즉 0일 경우는 뉴라벨을 없는걸로저장을 합니다.
    for i in cluster_case:
      if len(actual_case) > 0:
        idx = cluster_labels == i
        new_label=scipy.stats.mode(actual_labels[idx])[0][0]
        print(actual_case, "-",new_label) # 어떤 새로운 라벨과 매칭이 되는지에 대해서 보는것
        if new_label in actual_case:
            actual_case.remove(new_label) # 새로운 라벨이 actualcase에 있다하면 한번 제거를 해줘야한다 이유는 다음번에 또 나오면 안되어서 제거 하고 그렇치 않을경우에는
            #new 라벨을 temp라벨즈에 가져온다.
        else:
          new_label = temp_labels[new_label]
          temp_labels.remove(new_label)
          # 매칭되는 실제 label명을 dict형태로 저장.
      matched_cluster[i] = new_label
    else:
        new_label = None
        print(f"훈련된 label명: {i} >> 가장 빈번한 실제 label명: {new_label}")
    return matched_cluster
    #i의 값을 new_label로 지정을 해줍니다.

# 훈련된 dbscan label을 데이터가 많은 순서로 정렬해봅시다.
dbscan_labels = moon_dbscan.labels_
dbscan_case_dict = dict((x,list(dbscan_labels).count(x)) for x in set(dbscan_labels))
sorted_dbscan_case = sorted(dbscan_case_dict, key=dbscan_case_dict.get, reverse=True) #정렬은 sorted로 가능하다. reverse는 오름차순 내림차순
print(sorted_dbscan_case)

dbscan_perm_dict = find_matching_cluster(sorted_dbscan_case, moon_labels, dbscan_labels)
print(dbscan_perm_dict)

# 훈련된 label명과 실제 label명이 매칭되는 경우 >> 새로 매칭된 label명으로 변경.
# 훈련된 label명과 실제 label명이 매칭되지 않는 경우 >> 훈련된 label명 유지.

dbscan_new_labels = [label if label not in dbscan_perm_dict else dbscan_perm_dict[label] for label in dbscan_labels]
print(np.array(dbscan_new_labels[:80]))
print(np.array(dbscan_labels[:80]))

kmean_labels = moon_km.labels_
km_case = list(set(kmean_labels))
kmean_perm_dict = find_matching_cluster(km_case,moon_labels,kmean_labels)
kmean_perm_dict


kmean_new_labels = [kmean_perm_dict[label] for label in kmean_labels]
'''위와 다르게 개수를 알고있기때문에 그냥 진행을 해도 됩니다. '''
# 위에서 훈련된 k-means의 정확도(accuracy)를 계산하세요.
moon_kmeans_acc = accuracy_score(moon_labels, kmean_new_labels)
print(f"Accuracy score of K-means : {round(moon_kmeans_acc, 4)}")

# 위에서 훈련된 dbscan의 정확도(accuracy)를 계산하세요.
moon_dbscan_acc = accuracy_score(moon_labels, dbscan_new_labels)
print(f"Accuracy score of DBSCAN : {round(moon_dbscan_acc, 4)}")


'''문제 28. Clustering : DBSCAN - silhouette score 비교하기.
: 알고리즘에 따른 모델 평가 지표를 비교해보세요.'''
# silhouette score도 accuracy score과 동일한 결과인지 비교해보세요.
# 위에서 훈련된 k-means의 silhouette score을 계산하세요.
km_sc_value = silhouette_score(np.array(moon_data), kmean_new_labels, metric="euclidean", sample_size=None, random_state=None)
print(f'Silhouette score of K-means: {round(km_sc_value,4)}')

# 위에서 훈련된 dbscan의 silhouette score을 계산하세요.
dbscan_sc_value = silhouette_score(np.array(moon_data), dbscan_new_labels, metric="euclidean", sample_size=None, random_state=None)
print(f'Silhouette score of DBSCAN: {round(dbscan_sc_value,4)}')

'''slihouette의 결과를 보시면 위의 결과와 다르게 k-mean의 결과가 더 높습니다 그러한 이유는 
silhouette score 계산 방법의 특징 때문에, 
구형이 아닌 경우, 실제 시각화에서는 dbscan의 군집이 합리적이게 보이나, 
silhouette score 점수는 오히려 낮게 나올 수 있음.  
(데이터 분포가 구형이 아닌경우, 각 군집의 중심점이, 다른 군집과 가까워질 수 있기 때문.) 
두개를 같이 한는 이유가 데이터 분포에 따라가 해석이 달라져서 같이 비교하면 좋습니다. 
분포 클러스티링에서 db스캔처럼 잡힌것은 군집과 군집간의 분포 클러스티링을 하다보면 가까워져서 실루엣 스코어 점수가 낮게 나올수있습니다
수치적으로 비교할때는 참고하는게 좋습니다. 





문제 29. Clustering : DBSCAN - Adjusted rand index 비교하기.
: 비교할 label이 있는 경우, ARI로 수치적 비교가 가능. silhouette과는 달리, ARI는 유사도 계산에서 실제 label과 예측된 label을 비교하게 됩니다.
'''
from sklearn.metrics import adjusted_rand_score
km_ari = adjusted_rand_score(moon_labels, kmean_new_labels)
print(f'Adjusted rand index (ARI) of K-means: {round(km_ari,4)}')
dbscan_ari = adjusted_rand_score(moon_labels, dbscan_new_labels)
print(f'Adjusted rand index (ARI) of DBSCAN: {round(dbscan_ari,4)}')

'''dbscan을 사용할때는 accuracy를 한번 보고 모듈을 가져와서 ari를 확인해보는것을 추천한다.


[참고] Clustering : DBSCAN - 항상 DBSCAN이 best일까요?
'''

# iris 데이터에도 DBSCAN이 적합한지 비교해봅시다.
iris_compare_df = train_x.copy()

#k-means 훈련시키기
km_iris=KMeans(n_clusters=3).fit(iris_compare_df)
iris_compare_df["km_iris_label"]= km_iris.labels_
print(f"k-means label 종류: {list(set(km_iris.labels_))}")
#dbscan 훈련하기
dbscan_iris = DBSCAN(eps=0.05,min_samples=2).fit(iris_compare_df)
iris_compare_df["dbscan_iris+label"] = dbscan_iris.labels_
print(f"DBSCAN label 종류: {list(set(dbscan_iris.labels_))}")


# 수치로 비교하기.
# k-means.
iris_km_labels = km_iris.labels_ # km 훈련된 전체 lable.
iris_km_case = list(set(iris_km_labels)) # km 훈련된 lable 종류.
# label 매칭 시키기.
iris_km_perm_dict = find_matching_cluster(iris_km_case, train_y, iris_km_labels)
iris_km_new_labels = [iris_km_perm_dict[label] for label in iris_km_labels]
# DataFrame에 컬럼 추가하기.
iris_compare_df["new_km_iris_label"] = iris_km_new_labels
# iris_compare_df["new_km_iris_label"] = iris_compare_df["new_km_iris_label"].astype(str)
print(iris_km_perm_dict)

# dbscan.
iris_dbscan_labels = dbscan_iris.labels_ # dbscan 훈련된 전체 lable.
# labele 정렬하기.
iris_dbscan_case_dict = dict((x,list(iris_dbscan_labels).count(x)) for x in set(iris_dbscan_labels))
sorted_iris_dbscan_case = sorted(iris_dbscan_case_dict, key=iris_dbscan_case_dict.get, reverse=True)
iris_dbscan_case_dict

# label 매칭 시키기.
iris_dbscan_perm_dict = find_matching_cluster(sorted_iris_dbscan_case, train_y, iris_dbscan_labels)
#
iris_dbscan_new_labels = [label if label not in iris_dbscan_perm_dict else iris_dbscan_perm_dict[label] for label in iris_dbscan_labels]
# DataFrame에 컬럼 추가하기.
iris_compare_df["new_dbscan_iris_label"] = iris_dbscan_new_labels
# iris_compare_df["new_dbscan_iris_label"] = iris_compare_df["new_dbscan_iris_label"].astype(str)

# k-means의 정확도를 계산하세요.
kmeans_train_acc = accuracy_score(train_y, iris_km_new_labels)
print(f"Accuracy score of K-means train set : {round(kmeans_train_acc, 4)}")

# dbscan의 정확도를 계산하세요.
dbscan_iris_acc = accuracy_score(train_y, iris_dbscan_new_labels)
print(f"Accuracy score of DBSCAN : {round(dbscan_iris_acc, 4)}")
# 시각화로 비교하기.

fig = make_subplots(rows=1, cols=3, subplot_titles=("Actual", "K-means cluster", "DBSCAN cluster"))
# actual.
fig.add_trace(
    go.Scatter(x=iris_compare_df["sepal_width"],
               y=iris_compare_df["sepal_length"],
               mode="markers",
               marker=dict(color=train_y),
               text=train_y
               ),
    row=1, col=1
)
# k-means.
fig.add_trace(
    go.Scatter(x=iris_compare_df["sepal_width"],
               y=iris_compare_df["sepal_length"],
               mode="markers",
               marker=dict(color=iris_compare_df["new_km_iris_label"]),
               text=iris_compare_df["new_km_iris_label"]
               ),
    row=1, col=2
)
# dbscan.
fig.add_trace(
    go.Scatter(x=iris_compare_df["sepal_width"],
               y=iris_compare_df["sepal_length"],
               mode="markers",
               marker=dict(color=iris_compare_df["new_dbscan_iris_label"]),
               text=iris_compare_df["new_dbscan_iris_label"]
               ),
    row=1, col=3
)
fig.update_layout(height=600, width=1000, showlegend=False)
fig.show()


'''문제 30. Clustering : HDBSCAN - 다양한 분포/사이즈의 데이터 생성하기.'''
'''HDBSCAN 은 육안으로도 계층적 구조를 가지고 있는 여러가지 알고리즘을 섞여있어서 자기만의 장점이 있다.

'''
from sklearn.datasets import make_blobs

# HDBSCAN을 훈련시킬 데이터를 생성해보세요.
'''분포도를 선택을 해줄수있는데 중심점에 나눠서 설정하는게 아니고 한개의 분포도를 설정할수있어서 
 다른 분포를 가진 것을 생성하고 싶으면 하나 더 적어주어야합니다 그래서 blobs1,s2,s3 이렇게 지정을 해주어야합니다. 
 숫자가 커질수록 분포가 당연히 넓어집니다. 
 데이터를 한곳에 모으려면 numpy에서 제공하는 vstack을 쓰면 array를 list형태로 넣어주면 합쳐줍니다. np.vstack([moons,blobs1,s2,s3]_
   moon 데이터 형식이면 태극 모양으로 나타낼거고 중심점을 보면 답이 나온다 
   클러스터가 커지면 분포도가 넓어진다. 
    '''
moons, _ = make_moons(n_samples=100, noise=0.05)
blobs1, _ = make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
blobs2, _ = make_blobs(n_samples=30, centers=[(-0.3,-1), (4.0, 1.5)], cluster_std=0.3)
blobs3, _ = make_blobs(n_samples=100, centers=[(3,-1), (4.0, 1.5)], cluster_std=0.4)

hdb_data = np.vstack([moons, blobs1, blobs2, blobs3])
hdb_data_df = pd.DataFrame(hdb_data, columns=["x", "y"])
hdb_data_df.head()
# scatter plot 생성.
fig = px.scatter(hdb_data_df, x="x", y="y")
# 그래프 사이즈 조절.
fig.update_layout(width=600, height=500, title="HDBSCAN 데이터 분포")
# 그래프 확인.
fig.show()


''''문제 31. Clustering : HDBSCAN - HDBSCAN 알고리즘 탐색하기.
: 사용할 모듈을 install & import 하고, HDBSCAN 알고리즘의 파라미터를 살펴봅니다.

HDBSCAN 모듈 페이지 바로가기 >>

[Parameters]
 dbscan은 데이터 수를 지정해주었다면 여기는 조금 다르다. 
- min_cluster_size (default=5): 군집화를 위한 최소한의 cluster 사이즈.
- min_samples (default=None) : 반경내 있어야할 최소 data points.
- cluster_selection_epsilon(default=0.0): 거리 기준. 이 기준보다 아래의 거리는 cluster끼리 merge 됨.
트리 형태의 그래프를 그릴려면 거리의 기준점을 0을 주어야 그 이하의 것들은 merge가 되기 때문입니다. 

'''

# hdbscan 모듈을 import 합니다.
import hdbscan
hdbscan_model = hdbscan.HDBSCAN()
# help(hdbscan_model)
'''문제 32. Clustering : HDBSCAN - HDBSCAN 알고리즘 훈련시키기.

만든 데이터로 임의로 트레인을 하고 실행을 해보자 
'''
# HDBSCAN의 파라미터인 min_cluster_size 설정해봅시다.
# min_cluster_size=5
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5)

hdbscan_model.fit(hdb_data)
# 훈련된 결과 label을 확인해보세요.
hdbscan_label = hdbscan_model.fit_predict(hdb_data)
hdbscan_label[:10]
# 분류된 label의 총 갯수를 확인해보세요.
set(hdbscan_label)
'''문제 33. Clustering : HDBSCAN - HDBSCAN 알고리즘 파라미터 비교하기.
: min_cluster_size, min_samples, cluster_selection_epsilon 파라미터를 변화하면서 훈련된 label에 어떤 영향을 주는지 확인해봅시다.
'''
hdb_data_df["hdbscan_label"] = hdbscan_label
hdb_data_df["hdbscan_label"] = hdb_data_df["hdbscan_label"].astype(str)
fig = px.scatter(hdb_data_df, x="x", y="y", color="hdbscan_label")
fig.update_layout(width=600, height=500)
fig.show()

for mcn in [3,6,7,9,13]:
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=mcn,min_samples=None,predict_data=True).fit_predict(hdb_data)
    hdb_data_df["hdbscan_label"] = hdbscan_model
    hdb_data_df["hdbscan_label"] = hdb_data_df["hdbscan_label"].astype(str)
    # outlier 추세확인.
    hdbscan_case_dict = dict((x, list(hdbscan_label).count(x)) for x in set(hdbscan_label))
    outliers = hdbscan_case_dict[-1]

    fig = px.scatter(hdb_data_df, x="x", y="y", color="hdbscan_label")
    fig.update_layout(width=600, height=500,
                      title=f"min_cluster_size={mcn} > label수: {len(set(hdbscan_label))}, outlier: {outliers}")
    fig.show()
    '''최소 클러스터의 사이즈가 커졋다는 얘기는 label의 수가 줄어드는것이다 이유는 
    군집간의 분포도를 크게크게 잡아가기때문에 한 중심에 클러스터의 들어가는 양이 많아지면 즉 한 원안에 들어가는 양이 많아지면
    원의 갯수는 줄어들것입니다. <----내 생각 
    (강사왈: 클러스터 갯수가 적어지는것  한 파라미터의 사이즈가 이정도는 되야한다 즉 최소 사이즈가 커졌다는것은 한개의 클러스터 사이즈로 미달이라서
    옆에 있는것들도 한개의 군집으로 만들어지는것 
    '''
    # min_samples 파라미터가 3,5,7,9,13일때 clusters의 차이를 비교해보세요.
    for ms in [3, 5, 7, 9, 13]:
        hdbscan_label = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=ms, prediction_data=True).fit_predict(hdb_data)
        hdb_data_df["hdbscan_label"] = hdbscan_label
        hdb_data_df["hdbscan_label"] = hdb_data_df["hdbscan_label"].astype(str)

        # outlier 추세확인.
        hdbscan_case_dict = dict((x, list(hdbscan_label).count(x)) for x in set(hdbscan_label))
        outliers = hdbscan_case_dict[-1]

        fig = px.scatter(hdb_data_df, x="x", y="y", color="hdbscan_label")
        fig.update_layout(width=600, height=500,
                          title=f"min_samples={ms} > label수: {len(set(hdbscan_label))}, outlier: {outliers}")
        fig.show()

        '''min_sample도 확인이 가능합니다. 
        '''

        # cluster_selection_epsilon 파라미터가 0.1,0.5,0.7,1.0 일때 clusters의 차이를 비교해보세요.
        for cse in [0.1, 0.5, 0.7, 1.0]:
            hdbscan_label = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=cse,
                                            prediction_data=True).fit_predict(hdb_data)
            hdb_data_df["hdbscan_label"] = hdbscan_label
            hdb_data_df["hdbscan_label"] = hdb_data_df["hdbscan_label"].astype(str)

            # outlier 추세확인.
            hdbscan_case_dict = dict((x, list(hdbscan_label).count(x)) for x in set(hdbscan_label))
            outliers = hdbscan_case_dict[-1]

            fig = px.scatter(hdb_data_df, x="x", y="y", color="hdbscan_label")
            fig.update_layout(width=600, height=500,
                              title=f"cluster_selection_epsilon={cse} > label수: {len(set(hdbscan_label))}, outlier: {outliers}")
            fig.show()

'''문제 34. Clustering : HDBSCAN - HDBSCAN의 다양한 시각화 확인하기.
: HDBSCAN 알고리즘에서 사용되는 시각화를 해석해보세요.
실제 업무보다는 그냥 알아두면 좋다. 

'''

# [문제 33]에서 최적으로 판단되는 파라미터를 사용해 hdbscan 모델을 훈련시켜보세요.
# [참고] 시각화 생성을 위해 gen_min_span_tree=True로 훈련시켜야 합니다.
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.1, gen_min_span_tree=True).fit(hdb_data)

# 훈련된 모델을 사용해서 minimum_spanning_tree 를 생성해보세요.
# 각 point를 이어주는 line을 distance를 점수화한 mutual reachabillity를 사용하여 나타낸 그래프입니다.
# point간의 거리를 나타낸 것이 아닌, line은 그려나가면서 아직 추가되지 않은 point들 중에서 mutual reachabillity가 가장 낮은 point를 하나씩만 추가하는 방식으로 진행.
hdbscan_model.minimum_spanning_tree_.plot(edge_cmap="viridis",
                                      edge_alpha=0.9,
                                      node_size=10,
                                      edge_linewidth=1)
# 훈련된 모델을 사용해서 condensed_tree 를 생성해보세요.
# [참고] cluster도 함께 보기위해 select_clusters=True로 설정해주세요.
# 가장 오래 버틴 cluster 순으로 cluster을 분류합니다.
hdbscan_model.condensed_tree_.plot(select_clusters=True)


'''문제 35. Clustering : HDBSCAN - HDBSCAN와 K-means의 성능 비교하기.'''
# 7개의 그룹으로 k-means를 훈련시켜보세요.
hdb_data_km = KMeans(n_clusters=7).fit(hdb_data)
hdb_data_km
# 최적의 파라미터로 hdbscan를 훈련시켜보세요.
hdb_data_hdbscan_label = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.1, gen_min_span_tree=True).fit_predict(hdb_data)
hdb_data_hdbscan_label[:10]
hdb_data_df["kmeans_label"] = hdb_data_km.labels_
hdb_data_df["kmeans_label"] = hdb_data_df["kmeans_label"].astype(str)
hdb_data_df["hdbscan_label"] = hdb_data_hdbscan_label
hdb_data_df["hdbscan_label"] = hdb_data_df["hdbscan_label"].astype(str)

for label_case in ["hdbscan_label", "kmeans_label"]:
  fig = px.scatter(hdb_data_df, x="x", y="y", color=label_case)
  fig.update_layout(width=600, height=500, title=f"{label_case} 시각화")
  fig.show()

  '''문제 36. Clustering : HDBSCAN - HDBSCAN와 DBSCAN의 성능 비교하기.
  '''

hdb_data_dbscan = DBSCAN(eps=0.3, min_samples=5).fit(hdb_data)
hdb_data_df["dbscan_label"] = hdb_data_dbscan.labels_
hdb_data_df["dbscan_label"] = hdb_data_df["dbscan_label"].astype(str)

for label_case in ["hdbscan_label", "dbscan_label"]:
  fig = px.scatter(hdb_data_df, x="x", y="y", color=label_case)
  fig.update_layout(width=600, height=500, title=f"{label_case} 시각화")
  fig.show()
'''
문제 37. Clustering : HDBSCAN - 데이터 분포에 따른 HDBSCAN와 DBSCAN의 차이 확인하기.
: 데이터의 분산차이와 points 수 차이에 따른 알고리즘 성능의 차이를 확인해보세요.'''

# HDBSCAN와 DBSCAN을 비교할 데이터를 생성해보세요.
# [참고] 분산이 극단적인 두가지 케이스를 생성해봅니다.
blobs1, _ = make_blobs(n_samples=200, centers=[(-10, 5), (0, -5)], cluster_std=0.5)
blobs2, _ = make_blobs(n_samples=200, centers=[(30, -1), (30, 1.5)], cluster_std=5.0)

comp_data = np.vstack([blobs1, blobs2])
comp_data_df = pd.DataFrame(comp_data, columns=["x", "y"])

# scatter plot 생성.
fig = px.scatter(comp_data_df, x="x", y="y")
# 그래프 사이즈 조절.
fig.update_layout(width=600, height=500, title="데이터 분포")
# 그래프 확인.
fig.show()

# 생성된 데이터를 사용하여 dbscan와 hdbscan을 훈련시켜보세요.
# 시각화를 위해, 각 모델의 label을 dataframe에 저장하고 string으로 변환하여주세요.

# dbscan를 훈련시켜보세요.
dbscan_model = DBSCAN(eps=0.6, min_samples=10).fit(comp_data)
comp_data_df["dbscan_label"] = dbscan_model.labels_
comp_data_df["dbscan_label"] = comp_data_df["dbscan_label"].astype(str)

# hdbscan를 훈련시켜보세요.
hdbscan_lables = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.1, gen_min_span_tree=True).fit_predict(comp_data)
comp_data_df["hdbscan_label"] = hdbscan_lables
comp_data_df["hdbscan_label"] = comp_data_df["hdbscan_label"].astype(str)

# 시각화하기 이전에, outlier를 구분하기 위한 color 컬럼을 생성해주세요.
# [참고] 아래 color_dict를 사용해주세요.
color_dict = {"-1":"#d8d8d8", "0":"#ff5e5b", "1":"#457b9d", "2":"#00cecb", "3":"#FFED66"}
comp_data_df["dbscan_label_color"] = comp_data_df["dbscan_label"].map(color_dict)
comp_data_df["hdbscan_label_color"] = comp_data_df["hdbscan_label"].map(color_dict)

# 두 모델 결과를 시각화로 나타내고 차이가 나타나는지 확인해보세요.
# [참고] 회색으로 나타나는 point는 outlier로 분류된 points입니다.
for label_case in ["hdbscan_label", "dbscan_label"]:
  fig = go.Figure(data=go.Scatter(
      x=comp_data_df["x"],
      y=comp_data_df["y"],
      mode="markers",
      marker=dict(color=comp_data_df[label_case+"_color"], showscale=True)
  ))
  fig.update_layout(width=600, height=500, title=f"{label_case} 시각화")
  fig.show()

  '''분포가 극명하기에 아웃라이어를 dbscan_label을 많이 잡고있습니다.
   반면에 hdbscan 은 분포가 달라도 어느정도 잡아내는 알고리즘 입니다. '''

  '''label이 정해지지 않은 데이터의 분류 목적으로 사용되는 만큼, 모델 선택부터 평가까지 자유도가 높은 편입니다.
다양한 알고리즘의 차이를 기억하고, 데이터에 적합한 알고리즘을 사용하여 비교하는 것이 중요합니다.
군집화의 목적에 따라 평가 지표를 자유롭게 조절하는 것이 중요합니다.'''
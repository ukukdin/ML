'''문제 23. Clustering : DBSCAN - DBSCAN 알고리즘 탐색하기.
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
import plotly.express as px
from kneed import KneeLocator
from yellowbrick.cluster import SilhouetteVisualizer
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
import scipy
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer
import sys
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
# help(DBSCAN)

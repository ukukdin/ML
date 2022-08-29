'''iris 데이터로 기본적인 군집분석( clustering) 원리와 분석 과정을 연습해보자.

step 0: 군집분석의 기본지식 및 학습목표 확인

군집분석이란?  unsupervised learning 으로 데이터 간의 유사도를 정의하고 그 유사도에 가까운 것부터 순서대로 합쳐가는 방법이며,
유사도의 정의에는 거리나 상관계수 등 여러가지가 잇다.

            '학습목표'
군집분석(Clustering)의 다양한 알고리즘을 경험.
- 알고리즘 훈련에 필요한 파라미터를 알맞게 조절.
- 알고리즘 훈련 결과를 시각화, 수치화를 통해 평가.
- 서로 다른 알고리즘의 차이점을 이해.

'''

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', None)

# 데이터 url.
user_metadata_url = "https://raw.githubusercontent.com/dajeong-lecture/raw_data/main/user_meta_data.csv"
user_statsdata_url = "https://raw.githubusercontent.com/dajeong-lecture/raw_data/main/user_stats_data.csv"

# 위의 url를 사용해서 각각의 DataFrame을 생성해보세요.
user_metadata_df = pd.read_csv(user_metadata_url, parse_dates=[0])
user_stats_df = pd.read_csv(user_statsdata_url, parse_dates=[0])

# load된 데이터의 형태(shape)을 확인해보세요.
user_metadata_df.shape, user_stats_df.shape,
'''user id 와 기타 등등의 컬럼등이 존재한다. '''
print(user_metadata_df.head())
''' 유저의 고객 통계 데이터 구매 관련된 유저 사이드에 데이터 입니다.'''
print(user_stats_df.head())

'''
문제 2. E-Commerce 고객 메타데이터 둘러보기 (metadata)
: 데이터별 컬럼 명세서를 통해 E-Commerce 고객 메타 데이터를 파악해보세요.

[User metadata 컬럼 명세서]

컬럼명	설명
user_id	고객 고유 ID
sex	성별 (0:남, 1:여)
birthday_year	생일년도 (yyyy)
membership_type	멤버십 타입 (100:비회원, 300-500:정회원)
category_prefer	카테고리 선호 입력수 (int)
joined_date	가입일자 (yyyy-mm-dd)
deleted_date	탈퇴일자 (yyyy-mm-dd)
join_path	가입경로 (None:일반가입,1:sns가입)
os	os 타입 (IOS, AOS)
recommended_cnt	친구추천 수 (int)
'''
# 컬럼 데이터 타입을 확인해보세요
print(user_metadata_df.dtypes)


'''
 문제 3. E-Commerce 고객 메타데이터 전처리하기 (metadata)
: E-Commerce 고객 메타 데이터를 분석에 필요한 컬럼을 생성해보세요.
 '''

'''생일년도를 사용해서 2021년 기준 고객의 나이정보를 생성해보세요.'''
this_year =2022
user_metadata_df["age"] =this_year - user_metadata_df["birthday_year"] +1
print(user_metadata_df.head(2))

# 가입일자를 사용해서 2021년 1월 1일 기준 고객의 가입기간을 생성해보세요.
today_ymd = datetime(2022,1,1)
''' 가입일자의 포멧을 날짜형식('datetime64[ns]')으로 변경해보세요.'''
user_metadata_df["joined_data"] = user_metadata_df["joined_data"].astype['datatime64[ns]']
# 가입기간을 일(day)로 계산해보세요.
user_metadata_df["data_from_joined"] =(today_ymd - user_metadata_df["joined_data"]).dt.days
'''새로운 컬럼을 하나 생성하시고 위에 가입기간을 지정한 날짜에서 조인한 날짜를 빼고 .df.days 를 하면 날짜만 나오게할수있습니다.'''
# 가입기간을 년도(year)로 계산해보세요.
# [참고] np.timedelta64(1, "Y") 를 사용해보세요.
'''년도의 일수를 나눠주면 몇년인지 나옵니다. np.timedelta64는 365를 1로 포현하고 y로 적는다'''
user_metadata_df["years_from_joined"] = (today_ymd - user_metadata_df["joined_data"])/np.timedelta64(1,"Y")
'''소수점이 나오기때문에 round로 한번 수정을 해줍니다. 소수점 한자리까지만 보여주게끔.  '''
user_metadata_df["years_from_joined"] = user_metadata_df["years_from_joined"].round(1)
user_metadata_df.head(2)
'''

	user_id	sex	birthday_year	membership_type	category_prefer	joined_date	deleted_date	join_path	os	recommended_cnt	age	days_from_joined	years_from_joined
0	KjIRvUKVTgxGaek	0	1995	300	3	2019-09-09	NaT	NaN	AOS	4	27	480	1.3
1	QWhJIG1fOkhUJzG	1	1995	300	2	2020-04-13	NaT	NaN	AOS	1	27	263	0.7
 탈퇴일자를 사용해서 2021년 1월 1일 기준 고객의 탈퇴여부를 binary로 나타내보세요.
 [참고] 탈퇴일자는 string 컬럼입니다. 

# user_metadata_df[user_metadata_df["deleted_date"]=="NaT"]
# 1: 탈퇴함, 0: 탈퇴하지 않음
탈퇴를 햇는지 여부의 컬럼을 만들고 deleted_data 컬럼에서 스트링으로 되어있ㄴ느  NaT가 있는지 없는지 여부를 0과 1로 표시를 합니다.
'''
user_metadata_df["if_deleted"] = np.where(user_metadata_df["deleted_date"]=="NaT",0,1)


'''
문제 4. E-Commerce 고객 거래데이터 둘러보기 (transaction data)
: 데이터별 컬럼 명세서를 통해 E-Commerce 고객 거래 데이터를 파악해보세요.

[User stat 컬럼 명세서]
'''
# 아래 코드로 컬럼의 데이터 타입을 확인해보세요.
user_stats_df.dtypes

'''문제 5. E-Commerce 고객 거래데이터 전처리하기 (transaction data)
# 마지막 거래일자의 범위를 확인해 보세요.
'''
print(user_stats_df.last_data.min(),user_stats_df.last_date.max())
# 마지막 거래일자의 가장 오래된 날짜를 선택하여, Recency의 정도를 알아보는 컬럼을 생성해보세요.
# [참고] 기준점이 마지막 거래일자 직전일이 됩니다. (0일이 생기지 않게 하기위해)
oldest_ymd =datetime(2019,12,31)
'''oldest_ymd 를 기준으로 2020년 제일 max 날짜로 구매한 사람은 2019년12월31일 이후부터 제일 최근 주문날짜 안에 거래내역을 보여주고
2020-1-1주문이 마지막인 사람은 2019-12-31부터 해서 1월1일까지의 날짜를 보여주기 위해 제일 min한 날짜에서 -1일은 한 날짜로 계산하여
주문건수를 알려줍니다. 즉 주문 건수에서 0일이 나오지 않게 하는 방법입니다.'''



# 날짜컬럼으로 변경하고 Recency 컬럼을 생성해보세요. ('datetime64[ns]')
# 뒤에서 진행할 분석법에서 숫자가 클수록 높은 점수를 부여하기 위해 과거 시간을 기준으로 일수를 역산합니다.

user_stats_df["last_date"]= user_stats_df["last_date"].astype('datetime64[ns]')
user_stats_df["days_of_recency"] = (user_stats_df["last_date"] - oldest_ymd).dt.days


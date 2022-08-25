import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = dict()
df['2015'] = pd.read_csv('world_happiness_report/2015.csv')
df['2016'] = pd.read_csv('world_happiness_report/2016.csv')
df['2017'] = pd.read_csv('world_happiness_report/2017.csv')
df['2018'] = pd.read_csv('world_happiness_report/2018.csv')
df['2019'] = pd.read_csv('world_happiness_report/2019.csv')
df['2020'] = pd.read_csv('world_happiness_report/2020.csv')

'''Step 2. 데이터프레임 구성하기
문제 4. 년도별 데이터 표준화하기

'''

for key in df:
  print(key, df[key].columns) # 각 데이터프레임의 컬럼 확인

  # 각 년도별로 다른 정보를 가진 데이터 프레임의 Column을 동일하게 표준화하기
cols = ['country', 'score', 'economy', 'family', 'health', 'freedom', 'generosity', 'trust', 'residual']

df['2015'].drop(['Region', 'Happiness Rank', 'Standard Error'], axis=1, inplace=True)  # generosity, trust 순서 반대
df['2016'].drop(['Region', 'Happiness Rank', 'Lower Confidence Interval',
                   'Upper Confidence Interval'], axis=1, inplace=True)  # generosity, trust 순서 반대
df['2017'].drop(['Happiness.Rank', 'Whisker.high', 'Whisker.low'], axis=1, inplace=True)
df['2018'].drop(['Overall rank'], axis=1, inplace=True)  # residual 없음
df['2019'].drop(['Overall rank'], axis=1, inplace=True)  # residual 없음
df['2020'].drop(['Regional indicator', 'Standard error of ladder score',
                   'upperwhisker', 'lowerwhisker', 'Logged GDP per capita',
                   'Social support', 'Healthy life expectancy',
                   'Freedom to make life choices', 'Generosity',
                   'Perceptions of corruption', 'Ladder score in Dystopia'], axis=1, inplace=True)

df['2018']['residual'] = df['2018']['Score'] - df['2018'][['GDP per capita', 'Social support',
       'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']].sum(axis=1)
df['2019']['residual'] = df['2019']['Score'] - df['2019'][['GDP per capita', 'Social support',
       'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']].sum(axis=1)

print(df['2016'].columns)

df['2015'] = df['2015'][['Country', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)',
       'Dystopia Residual']]
df['2016'] = df['2016'][['Country', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)',
       'Dystopia Residual']]


for key in df:
  print(key, df[key].columns) # 각 데이터프레임의 컬럼 확인

for key in df:
    df[key].columns = cols

'''문제 5. 하나의 데이터프레임으로 합치기'''

# 아래 셀과 동일한 데이터프레임으로 결합하기
df_all = pd.concat(df, axis=0)
df_all.index.names = ['year', 'rank']
df_all


'''문제 6. 원하는 형태로 데이터프레임 정리하기'''

# 아래 셀과 동일한 데이터프레임으로 변형하기
df_all.reset_index(inplace=True)
df_all['rank'] += 1
df_all


'''문제 7. Pivot을 이용하여 데이터프레임 재구성하기'''

# 아래 셀과 동일한 데이터프레임 구성하기
# Hint) DataFrame의 pivot() 메소드 활용
rank_table = df_all.pivot(index='country', columns=['year'], values='rank')
rank_table.sort_values('2020', inplace=True)
rank_table.head(20)


'''Step 3. 데이터 시각화 수행하기
'''

'''문제 8. 년도별 순위 변화 시각화하기
'''

# 아래 셀과 동일하게 년도별 순위 변화를 시각화하기
# Hint) plt.plot을 이용하고, 필요한 경우 데이터프레임을 변형하면서 그리시오.

fig = plt.figure(figsize=(10, 50))
rank2020 = rank_table['2020'].dropna()
for c in rank2020.index:
  t = rank_table.loc[c].dropna()
  plt.plot(t.index, t, '.-')

plt.xlim(['2015', '2020'])
plt.ylim([0, rank_table.max().max() + 1])
plt.yticks(rank2020, rank2020.index)
ax = plt.gca()
ax.invert_yaxis()
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
plt.tight_layout()
plt.show()

'''문제 9. 분야별로 나누어 점수 시각화하기
'''

# sns.barplot()을 이용하여 아래 셀과 동일하게 시각화하기
# Hint) 필요에 따라 데이터프레임을 수정하여 사용하시오. 적절한 수정을 위해 누적합(pd.cumsum())을 활용하시오.
df_all

fig = plt.figure(figsize=(6, 8))
data = df_all[df_all['year'] == '2020']
data = data.loc[data.index[:20]]

d = data[data.columns[4:]].cumsum(axis=1)
d = d[d.columns[::-1]]
d['country'] = data['country']

sns.set_color_codes('muted')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'purple'][::-1]
for idx, c in enumerate(d.columns[:-1]):
  sns.barplot(x=c, y='country', data=d, label=c, color=colors[idx])

plt.legend(loc='lower right')
plt.title('Top 20 Happiness Scores in Details')
plt.xlabel('Happiness Score')
sns.despine(left=True, bottom=True)

'''문제 10. Column간의 상관성 시각화하기'''
# 상관성 Heatmap, Pairplot 등으로 상관성을 시각화하기
sns.heatmap(df_all.drop('rank', axis=1).corr(), annot=True, cmap='YlOrRd')

sns.pairplot(df_all.drop('rank', axis=1))

'''Step 4. 모델 학습을 위한 데이터 전처리
11. 모델의 입력과 출력 정의하기 
'''
# 학습할 모델의 입출력을 정의하시오. Column의 의미를 고려하여 선정하시오.
col_input_list = ['economy','family','health','freedom','generosity','trust']
col_out = 'score'


'''문제 12. 학습데이터와 테스트데이터 분리하기
# 2015년 ~ 2019년도 데이터를 학습 데이터로, 2020년도 데이터를 테스트 데이터로 분리하기
'''
df_train = df_all[df_all['year']!=['2020']]
df_test = df_all[df_all[col_out]]

X_train = df_train[col_input_list]
y_train = df_train[col_out]
X_test = df_test[col_input_list]
y_test = df_test[col_out]


'''문제 13. StandardScaler를 이용해 학습 데이터 표준화하기'''
from sklearn.preprocessing import StandardScaler
# StandardScaler를 이용해 학습 데이터를 표준화하기
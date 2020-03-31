import sys
import pickle
import numpy as np
import pandas as po
import matplotlib.pyplot as plt

from gan import output
sys.modules["output"] = output
from gan.output import OutputType
from gan.output import Normalization

df = po.read_csv('data/processed/energy.csv').drop('Unnamed: 0', axis = 1)
df.head()

df['Month'] = df['date'].apply(lambda x: x.split('/')[1])
df.head()

df = df[['Values', 'Day_of_week', 'Month', 'Day_of_month']]
df = po.get_dummies(df, columns = ['Day_of_week', 'Month', 'Day_of_month'])
df.head()

plt.plot(df['Values'])

# ['Day_of_week_0', 'Day_of_week_1', 'Day_of_week_2', 'Day_of_week_3', 'Day_of_week_4', 'Day_of_week_5', 'Day_of_week_6']

df = df[:len(df)-len(df)%7]
len(df)

df['Values'] = (df['Values'] - df['Values'].mean())/df['Values'].std()

df['Values'].std()

df['Values'].mean()

plt.plot(df['Values'])

feats = df[['Values', 'Day_of_week_0', 'Day_of_week_1', 'Day_of_week_2', 'Day_of_week_3', 'Day_of_week_4', 'Day_of_week_5', 'Day_of_week_6']]
attrs = df.drop(['Values', 'Day_of_week_0', 'Day_of_week_1', 'Day_of_week_2', 'Day_of_week_3', 'Day_of_week_4', 'Day_of_week_5', 'Day_of_week_6'], axis = 1)

data_feature_output = [
    output.Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE, is_gen_flag=False),
    output.Output(type_=OutputType.DISCRETE, dim=7, normalization=None, is_gen_flag=False)]

data_attribute_output = [
    output.Output(type_=OutputType.DISCRETE, dim=12, normalization=None, is_gen_flag=False),
    output.Output(type_=OutputType.DISCRETE, dim=31, normalization=None, is_gen_flag=False)]

# [(number of training samples) x (maximum length) x (total dimension of features)]

SEQ_LEN = 7

seqs_f = []
for i in range(0, len(feats), SEQ_LEN):
    values = []
    for j in range(len(feats[i:i+SEQ_LEN])):
        values.append(feats[i:i+SEQ_LEN].iloc[j].tolist())
    seqs_f.append(values)

data_feature = np.array(seqs_f, ndmin = 3)
data_feature.shape

#this will lead to slight errors at the end of months. Fix by padding later
seqs_a = []
for i in range(0, len(attrs), SEQ_LEN):
    seqs_a.append(attrs.iloc[i].tolist())

data_attribute = np.array(seqs_a)
data_attribute.shape

data_gen_flag = np.ones((data_feature.shape[0], SEQ_LEN))
data_gen_flag.shape

data_feature[-1]



with open('data/energy/data_train.npz', 'wb') as file:
    np.savez(file, data_feature, data_attribute, data_gen_flag)

with open('data/energy/data_test.npz', 'wb') as file:
    np.savez(file, data_feature, data_attribute, data_gen_flag)

with open('data/energy/data_feature_output.pkl', 'wb') as file:
    pickle.dump(data_feature_output, file)

with open('data/energy/data_attribute_output.pkl', 'wb') as file:
    pickle.dump(data_attribute_output, file)

with open('data/energy/data_train.npz', 'rb') as file:
    arrays = np.load(file)
    data_feature_l = arrays['arr_0']
    data_attribute_l = arrays['arr_1']
    data_gen_l = arrays['arr_2']



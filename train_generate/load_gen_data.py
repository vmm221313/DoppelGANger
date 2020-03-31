import numpy as np
import pandas as po
import matplotlib.pyplot as plt

with open('generated_samples/epoch_id-399/generated_data_train.npz', 'rb') as file:
    arrays = np.load(file)
    print(arrays.files)
    data_feature_l = arrays['data_feature']
    data_attribute_l = arrays['data_attribute']
    data_gen_l = arrays['data_gen_flag']

with open('data/energy/data_train.npz', 'rb') as file:
    arrays = np.load(file)
    data_feature_l_o = arrays['arr_0']
    data_attribute_l_o = arrays['arr_1']
    data_gen_l_o = arrays['arr_2']

values_o = []
for seq in data_feature_l_o:
    for row in seq:
        values_o.append(row[0])

values = []
for seq in data_feature_l:
    for row in seq:
        values.append(row[0])

plt.plot(values)

np.mean(values)

np.std(values)

len(values_o)

np.mean(values_o)

np.std(values_o)

# +
#1-week, 1-month etc. shifted correlations
# -



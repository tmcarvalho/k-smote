"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from kanon import single_outs
# %%
data, key_vars = single_outs()
# %% 
# check the target variable that is single out and not single out
data['single_out'].value_counts()
# 0 -> non single out
# 1 -> single out
# %% percentage of single outs
percentage = data['single_out'].value_counts()[1]*100 / data.shape[0]
print(percentage)
# %%
g = sns.countplot(data['single_out'])
g.set_xticklabels(['Not single out','Single out'])
plt.show()
# %%
smote = SMOTE(random_state=42, k_neighbors=3)
# fit predictor and target variable
X = data[key_vars]
y = data.iloc[:, -1]
x_smote, y_smote = smote.fit_resample(X, y)
# %%
#print('Original dataset shape', data['single_out'].value_counts())
print('Resample dataset shape', y_smote.value_counts())
# %%
print(f'{y_smote.value_counts()[1]-data["single_out"].value_counts()[1]} new observations included')
# %%

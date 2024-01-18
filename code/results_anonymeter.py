# %%
from os import walk
import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
from matplotlib import pyplot as plt

# %%
def concat_each_file(folder):
    _, _, input_files = next(walk(f'{folder}'))
    concat_results = pd.DataFrame()

    for file in input_files:
        risk = pd.read_csv(f'{folder}/{file}')
        risk['ds_complete']=file
        concat_results = pd.concat([concat_results, risk])

    return concat_results

# %%
risk_ppt = concat_each_file('../output/anonymeter/PPT_ARX')
# %%
risk_resampling= concat_each_file('../output/anonymeter/re-sampling')
# %%
risk_deeplearning= concat_each_file('../output/anonymeter/deep_learning')
# %%
risk_city = concat_each_file('../output/anonymeter/city')
# %%
risk_privateSMOTE = concat_each_file('../output/anonymeter/PrivateSMOTE')

# %%
risk_ppt['technique'] = 'PPT'
risk_resampling['technique'] = risk_resampling['ds_complete'].apply(lambda x: x.split('_')[1].title())
risk_deeplearning['technique'] = risk_deeplearning['ds_complete'].apply(lambda x: x.split('_')[1])
risk_city['technique'] = risk_city['ds_complete'].apply(lambda x: x.split('_')[1].upper())
risk_privateSMOTE['technique'] = r'$\epsilon$-PrivateSMOTE'

# %%
results = pd.concat([risk_ppt, risk_resampling, risk_deeplearning, risk_city, risk_privateSMOTE
                    ]).reset_index(drop=True)
# %%
results['ds'] = results['ds_complete'].apply(lambda x: x.split('_')[0])
# %%
results.loc[results['technique']=='Under', 'technique'] = 'RUS'
results.loc[results['technique']=='Bordersmote', 'technique'] = 'BorderlineSMOTE'
results.loc[results['technique']=='Smote', 'technique'] = 'SMOTE'
results.loc[results['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
# %% remove wrong results (dpgan in deep learning folders) 
results = results.loc[results.technique != 'dpgan'].reset_index(drop=True)

# %%
# results.to_csv('../output_analysis/anonymeter.csv', index=False)

# %%
# results = pd.read_csv('../output_analysis/anonymeter.csv')
# %%
order = ['PPT', 'RUS', 'SMOTE', 'BorderlineSMOTE', 'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN', r'$\epsilon$-PrivateSMOTE']
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results,
    x='technique', y='value', order=order, color='c')
sns.set(font_scale=2.5)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, title='', borderaxespad=0., frameon=False)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Privacy Risk (Linkability)")
plt.show()

# %%  BETTER IN PRIVACY
results_risk_min = results.loc[results.groupby(['ds', 'technique'])['value'].idxmin()].reset_index(drop=True)
sns.set_style("darkgrid")
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_risk_min,
    x='technique', y='value', order=order, color='c')
sns.set(font_scale=2.5)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, title='', borderaxespad=0., frameon=False)
#ax.set_yscale("symlog")
#ax.set_ylim(-0.2,150)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Privacy Risk (Linkability)")
plt.show()


# %%
privsmote = results.loc[results.technique.str.contains('PrivateSMOTE')].reset_index(drop=True)
privsmote['epsilon'] = np.nan
for idx, file in enumerate(privsmote.ds_complete):
    if 'privateSMOTE' in file:
        privsmote['epsilon'][idx] = str(list(map(float, re.findall(r'\d+\.\d+', file.split('_')[1])))[0])
        
# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

ep_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
sns.set_style("darkgrid")
plt.figure(figsize=(8,7))
ax = sns.boxplot(data=privsmote, x='epsilon', y='value', order=ep_order, **PROPS)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Privacy Risk (Linkability)")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/privateSMOTE_epsilons_risk.pdf', bbox_inches='tight')

# %%
privsmote_max = privsmote.loc[privsmote.groupby(['ds', 'epsilon'])['value'].idxmin()]

ep_order = ['0.1', '0.5', '1.0', '5.0', '10.0']
sns.set_style("darkgrid")
plt.figure(figsize=(8,7))
ax = sns.boxplot(data=privsmote_max, x='epsilon', y='value', order=ep_order, **PROPS)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Privacy Risk (Linkability)")
plt.show()
# %%

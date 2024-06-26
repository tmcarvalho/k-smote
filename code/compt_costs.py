
# %%
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import os
# %%
comp_costs_files = next(os.walk('../output/comp_costs'))[2]

# %%
all_costs = pd.DataFrame()
for file in comp_costs_files:
    dict= pd.read_json(f'../output/comp_costs/{file}')
    # df = pd.DataFrame.from_dict(dict, orient="index")
    all_costs = pd.concat([all_costs, dict])

# %%
summary_costs = all_costs.groupby('file').agg({'elapsed_time': 'max', 'cpu_percent': 'mean', 'gpu_percent': 'mean', 'ram_percent': 'mean', 'gpu_temperature': 'mean'})

# %%
summary_costs = summary_costs.reset_index()
# %%
summary_costs['technique'] = summary_costs.file.apply(lambda x: x.split('_')[1])
# %%
summary_costs['technique'] = [r'$\epsilon$-3PrivateSMOTE' if 'privateSMOTE3' in tech else tech for tech in summary_costs['technique']]
summary_costs['technique'] = [r'$\epsilon$-5PrivateSMOTE' if 'privateSMOTE5' in tech else tech for tech in summary_costs['technique']]
summary_costs['technique'] = [r'$\epsilon$-2PrivateSMOTE' if '-privateSMOTE' in tech else tech for tech in summary_costs['technique']]
# %%
summary_costs.loc[summary_costs['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
summary_costs.loc[summary_costs['technique']=='dpgan', 'technique'] = 'DPGAN'
summary_costs.loc[summary_costs['technique']=='pategan', 'technique'] = 'PATE-GAN'

# %%
# summary_costs.to_csv('../output_analysis/comp_costs.csv', index=False)
# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
# %% Remove ds32, 33 and 38 because they do not have borderline and smote
pattern_to_exclude = re.compile(r'ds3[238]')
summary_costs = summary_costs[~summary_costs.file.str.contains(pattern_to_exclude)]

# %% remove ds55
summary_costs = summary_costs[~summary_costs.file.str.contains('ds55')]

# %% remove epsilon=10
epi_to_exclude = re.compile(r'epi10.0')
summary_costs = summary_costs[~summary_costs.file.str.contains(epi_to_exclude)]
# %%
order = ['Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATE-GAN', r'$\epsilon$-2PrivateSMOTE', r'$\epsilon$-3PrivateSMOTE', r'$\epsilon$-5PrivateSMOTE']
sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=summary_costs["technique"], y=summary_costs["elapsed_time"], order=order,**PROPS)
# sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_ylim(0,1250)
ax.set_ylabel("Time (sec)")
ax.set_xlabel("")
sns.set(font_scale=1.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/time.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=summary_costs["technique"], y=summary_costs["cpu_percent"], order=order,**PROPS)
# sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_ylim(-1.5,100)
ax.set_ylabel("Percentage of CPU")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/cpu.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=summary_costs["technique"], y=summary_costs["gpu_percent"], order=order,**PROPS)
# sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_ylim(-1.5,100)
ax.set_ylabel("Percentage of GPU")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/gpu.pdf', bbox_inches='tight')
# %%
summary_costs['ds'] = summary_costs['file'].apply(lambda x: x.split('_')[0])
# %%
summary_costs_max = summary_costs.groupby(['ds', 'technique']).agg({'elapsed_time': 'sum', 'cpu_percent': 'mean', 'gpu_percent': 'mean', 'ram_percent': 'mean', 'gpu_temperature': 'mean'}).reset_index()

# %%
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=summary_costs_max["technique"], y=summary_costs_max["elapsed_time"], order=order,**PROPS)
# sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
# ax.set_ylim(-1.5,100)
ax.set_ylabel("Time")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# %%
summary_costs_final = summary_costs.groupby(['technique']).agg({'elapsed_time': 'sum', 'cpu_percent': 'mean', 'gpu_percent': 'mean', 'ram_percent': 'mean', 'gpu_temperature': 'mean'}).reset_index()

# %%
summary_costs_final['elapsed_time_min'] = summary_costs_final['elapsed_time'] / 60
# %%
summary_costs_final['n_variants'] = summary_costs.groupby(['technique']).size().reset_index(name='count')['count']

# %%
summary_costs_final['time_variant'] = summary_costs_final['elapsed_time_min'] / summary_costs_final['n_variants']

# %%

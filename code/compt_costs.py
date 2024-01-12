
# %%
import pandas as pd
import seaborn as sns
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
summary_costs['technique'] = ['PrivateSMOTE' if 'privateSMOTE' in tech else tech for tech in summary_costs['technique']]
# %%
summary_costs.loc[summary_costs['technique']=='PrivateSMOTE', 'technique'] = r'$\epsilon$-PrivateSMOTE'
summary_costs.loc[summary_costs['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
summary_costs.loc[summary_costs['technique']=='dpgan', 'technique'] = 'DPGAN'
summary_costs.loc[summary_costs['technique']=='pategan', 'technique'] = 'PATEGAN'

# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
# %%
order = ['Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN', r'$\epsilon$-PrivateSMOTE']

plt.figure(figsize=(12,10))
ax = sns.boxplot(x=summary_costs["technique"], y=summary_costs["elapsed_time"], order=order,**PROPS)
# sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_ylim(0,1250)
ax.set_ylabel("Time")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# sns.set(font_scale=1.7)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/time.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=summary_costs["technique"], y=summary_costs["cpu_percent"], order=order,**PROPS)
# sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_ylim(-1.5,100)
ax.set_ylabel("Percentage of CPU")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# sns.set(font_scale=1.7)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/cpu.pdf', bbox_inches='tight')

# %%
plt.figure(figsize=(12,10))
ax = sns.boxplot(x=summary_costs["technique"], y=summary_costs["gpu_percent"], order=order,**PROPS)
# sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformation', borderaxespad=0., frameon=False)
ax.set_ylim(-1.5,100)
ax.set_ylabel("Percentage of GPU")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# sns.set(font_scale=1.7)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/plots/gpu.pdf', bbox_inches='tight')

# %%

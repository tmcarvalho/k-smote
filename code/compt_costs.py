
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# %%
all_results = pd.read_csv('../output/predictiveresults.csv')
# %%
# tvae = all_results.loc[all_results['technique'] == 'TVAE']

# %%
grp_dataset = all_results.groupby('ds')
for name, grp in grp_dataset:
    # print(name)
    # print(grp['qi'].nunique())
    sns.set_style('darkgrid')
    sns.set(font_scale=1.5)
    g = sns.FacetGrid(grp,row='ds', height=8, aspect=1.5)
    g.map(sns.boxplot, "technique", "mean_test_f1_weighted_perdif","model",
    palette='muted').add_legend()
    plt.xticks(rotation=45)
    #g.set_axis_labels("Nearest Neighbours", "Fscore")
    g.set_titles("{row_name}")
    g.savefig(
      f'{os.path.dirname(os.getcwd())}/output/plots/each/{name}_fscore.pdf',
      bbox_inches='tight')
# %%
############### privateSMOTE
privatesmote = pd.read_csv('../PwrData_2022-9-21_20-7-17_privateSMOTEscratch.csv')
privatesmote_clean = privatesmote.loc[privatesmote['Elapsed Time (sec)'] < 1703]

privatesmote_clean['Elapsed Time (min)'] = privatesmote_clean['Elapsed Time (sec)']/60
privatesmote_clean['Elapsed Time (min)'] = privatesmote_clean['Elapsed Time (min)'].astype(int)
# %% Temperature
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)
# %% Processor power
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_clean, x="Elapsed Time (sec)", y="Processor Power_0(Watt)")
sns.set(font_scale=2)
# %% CPU Frequency
sns.lineplot(data=privatesmote_clean, x="Elapsed Time (sec)", y="CPU Max Frequency_0(MHz)")
# %%
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_clean, x="Elapsed Time (min)", y="CPU Utilization(%)")
sns.set(font_scale=2)
# %%
privatesmote_clean_temp = privatesmote_clean[['Elapsed Time (sec)', 'CPU Max Temperature_0(C)']]
privatesmote_clean_temp['Technique'] = 'privateSMOTE'
# %%
############## privateSMOTE scratch
privatesmote_scratch = pd.read_csv('../PwrData_2022-9-21_20-7-17_privateSMOTEscratch.csv')
privatesmote_scratch_clean = privatesmote_scratch.loc[privatesmote_scratch['Elapsed Time (sec)'] < 1720]

privatesmote_scratch_clean['Elapsed Time (min)'] = privatesmote_scratch_clean['Elapsed Time (sec)']/60
privatesmote_scratch_clean['Elapsed Time (min)'] = privatesmote_scratch_clean['Elapsed Time (min)'].astype(int)

# %%
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_scratch_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)
# %%
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_scratch_clean, x="Elapsed Time (min)", y="Processor Power_0(Watt)")
sns.set(font_scale=2)
# %% CPU Frequency
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_scratch_clean, x="Elapsed Time (sec)", y="CPU Max Frequency_0(MHz)")
sns.set(font_scale=2)
# %% CPU Utilization
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_scratch_clean, x="Elapsed Time (min)", y="CPU Utilization(%)")
sns.set(font_scale=2)
# %% Cumulative Energy
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_scratch_clean, x="Elapsed Time (min)", y="Cumulative Processor Energy_0(Joules)")
sns.set(font_scale=2)
# %%
############## TVAE
tvae = pd.read_csv('../PwrData_2022-9-22_20-15-55_TVAE.csv')
# %%
tvae_clean = tvae.loc[tvae['Elapsed Time (sec)'] < 19850]
# %%
tvae_clean['Elapsed Time (min)'] = tvae_clean['Elapsed Time (sec)']/60
tvae_clean['Elapsed Time (min)'] = tvae_clean['Elapsed Time (min)'].astype(int)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=tvae_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)

# %%
plt.figure(figsize=(25,12))
ax = sns.lineplot(data=tvae_clean, x='Elapsed Time (min)', y="Processor Power_0(Watt)")
# ax.set_xlim(tvae_clean["Elapsed Time (sec)"].min(),tvae_clean["Elapsed Time (sec)"].min() + 200)
sns.set(font_scale=2)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=tvae_clean, x="Elapsed Time (sec)", y="CPU Max Frequency_0(MHz)")
sns.set(font_scale=2)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=tvae_clean, x="Elapsed Time (sec)", y="CPU Utilization(%)")
sns.set(font_scale=2)
# %%
plt.figure(figsize=(20,12))
sns.lineplot(data=tvae_clean, x="Elapsed Time (min)", y="Cumulative Processor Energy_0(Joules)")
sns.set(font_scale=2)

# %% ############# CTGAN
ctgan = pd.read_csv('../PwrData_2022-9-23_7-26-51_CTGAN.csv')
# %%
ctgan_clean = ctgan.loc[ctgan['Elapsed Time (sec)'] < 57300]
# %%
ctgan_clean['Elapsed Time (min)'] = ctgan_clean['Elapsed Time (sec)']/60
ctgan_clean['Elapsed Time (min)'] = ctgan_clean['Elapsed Time (min)'].astype(int)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=ctgan_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)

# %% ############# CopulaGAN
copula_gan = pd.read_csv('../PwrData_2022-9-28_21-1-44_CopulaGAN.csv')
# %%
copula_gan_clean = copula_gan.loc[copula_gan['Elapsed Time (sec)'] < 56000]
# %%
copula_gan_clean['Elapsed Time (min)'] = copula_gan_clean['Elapsed Time (sec)']/60
copula_gan_clean['Elapsed Time (min)'] = copula_gan_clean['Elapsed Time (min)'].astype(int)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=copula_gan_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)

# %% ############# Gaussian Copula
gauss_copula = pd.read_csv('../PwrData_2022-9-30_21-11-28_GaussianCopula.csv')
# %%
gauss_copula_clean = gauss_copula.loc[gauss_copula['Elapsed Time (sec)'] < 49]
# %%
gauss_copula_clean['Elapsed Time (min)'] = gauss_copula_clean['Elapsed Time (sec)']/60
gauss_copula_clean['Elapsed Time (min)'] = gauss_copula_clean['Elapsed Time (min)'].astype(int)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=gauss_copula_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)

# %% ######################## JOIN ALL
copula_gan_clean['Technique'] = 'Copula GAN'
tvae_clean['Technique'] = 'TVAE'
ctgan_clean['Technique'] = 'CTGAN'
privatesmote_clean['Technique'] = 'privateSMOTE'
# %%
all = pd.concat([copula_gan_clean, tvae_clean, ctgan_clean, privatesmote_clean]).reset_index(drop=True)
# %%
plt.figure(figsize=(15,12))
g = sns.lineplot(data=all, x="Elapsed Time (min)", y="Cumulative Processor Energy_0(Joules)", 
    hue='Technique', sizes=(.5), alpha=0.7)
g.set_yscale("log")
sns.set(font_scale=2)
# %%
plt.figure(figsize=(15,12))
g = sns.lineplot(data=all, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)", hue='Technique',
    sizes=(.25))
#g.set_yscale("log")
sns.set(font_scale=2)
# %%
plt.figure(figsize=(20,12))
g = sns.lineplot(data=all, x="Elapsed Time (min)", y="Package Temperature_0(C)", hue='Technique',
    sizes=(.25))
#g.set_yscale("log")
sns.set(font_scale=2)
# %%

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# %%
############### privateSMOTE
privatesmote = pd.read_csv('../computational_costs/PwrData_2022-9-21_20-7-17_privateSMOTEscratch.csv')
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
privatesmote_clean['Cumulative Processor Energy_0(Joules)'].max()
# %%
privatesmote_clean_temp = privatesmote_clean[['Elapsed Time (sec)', 'CPU Max Temperature_0(C)']]
privatesmote_clean_temp['Technique'] = 'privateSMOTE'
# %%
############## privateSMOTE scratch
privatesmote_scratch = pd.read_csv('../computational_costs/PwrData_2022-9-21_20-7-17_privateSMOTEscratch.csv')
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
tvae = pd.read_csv('../computational_costs/PwrData_2022-9-22_20-15-55_TVAE.csv')
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
ctgan = pd.read_csv('../computational_costs/PwrData_2022-9-23_7-26-51_CTGAN.csv')
# %%
ctgan_clean = ctgan.loc[ctgan['Elapsed Time (sec)'] < 57300]
# %%
ctgan_clean['Elapsed Time (min)'] = ctgan_clean['Elapsed Time (sec)']/60
ctgan_clean['Elapsed Time (min)'] = ctgan_clean['Elapsed Time (min)'].astype(int)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=ctgan_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)
# %%
ctgan_clean['Elapsed Time (min)'].max()
# %%
ctgan_clean["Cumulative Processor Energy_0(Joules)"].max()
# %% ############# CopulaGAN
copula_gan = pd.read_csv('../computational_costs/PwrData_2022-9-28_21-1-44_CopulaGAN.csv')
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
gauss_copula = pd.read_csv('../computational_costs/PwrData_2022-9-30_21-11-28_GaussianCopula.csv')
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
################### PrivateSMOTE Force Laplace
privatesmote_force_laplace = pd.read_csv('../computational_costs/PwrData_2023-5-23_21-47-37_PrivateSMOTE_force_laplace.csv')
# %%
privatesmote_force_laplace_clean = privatesmote_force_laplace.loc[privatesmote_force_laplace['Elapsed Time (sec)'] < 7500]
# %%
plt.figure(figsize=(20,12))
sns.lineplot(data=privatesmote_force_laplace_clean, x="Elapsed Time (sec)", y="Processor Power_0(Watt)")
sns.set(font_scale=2)
# %%
privatesmote_force_laplace_clean['Elapsed Time (min)'] = privatesmote_force_laplace_clean['Elapsed Time (sec)']/60
privatesmote_force_laplace_clean['Elapsed Time (min)'] = privatesmote_force_laplace_clean['Elapsed Time (min)'].astype(int)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=privatesmote_force_laplace_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)
# %%
privatesmote_force_laplace_clean['Cumulative Processor Energy_0(Joules)'].max()
# %%
privatesmote_force_laplace_clean['Elapsed Time (min)'].max()

# %%
################### DPART
dpart = pd.read_csv('../computational_costs/PwrData_2023-6-2_14-33-55_dpart.csv')
# %%
dpart_clean = dpart.loc[dpart['Elapsed Time (sec)'] < 101]
# %%
plt.figure(figsize=(20,12))
sns.lineplot(data=dpart_clean, x="Elapsed Time (sec)", y="Processor Power_0(Watt)")
sns.set(font_scale=2)
# %%
dpart_clean['Elapsed Time (min)'] = dpart_clean['Elapsed Time (sec)']/60
dpart_clean['Elapsed Time (min)'] = dpart_clean['Elapsed Time (min)'].astype(int)
# %%
plt.figure(figsize=(25,12))
sns.lineplot(data=dpart_clean, x="Elapsed Time (min)", y="CPU Max Temperature_0(C)")
sns.set(font_scale=2)
# %%
dpart_clean['Cumulative Processor Energy_0(Joules)'].max()
# %%
dpart_clean['Elapsed Time (min)'].max()

# %% New results from VM
import pandas as pd

# Reading JSON data from a file
costs_5PS= pd.read_json("../output/comp_costs/5-PrivateSMOTE.json")

# %%

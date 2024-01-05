import pandas as pd
import numpy as np
import os
from sdmetrics.single_column import StatisticSimilarity, RangeCoverage, CategoryCoverage

orig_files = next(os.walk('original'))[2]
privsmote_files = next(os.walk('output/oversampled/PrivateSMOTE'))[2]

all_stats = []
for privsmote in privsmote_files:
    for orig in orig_files:
        f = int(orig.split('.csv')[0])
        if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
            if f'ds{f}' == privsmote.split('_')[0]:
                print(f)
                privsmote_data = pd.read_csv(f'output/oversampled/PrivateSMOTE/{privsmote}')
                privsmote_data = privsmote_data.loc[:, privsmote_data.columns[:-1]]
                orig_data = pd.read_csv(f'original/{orig}')
                num_columns = orig_data.select_dtypes(include=np.number).columns
                # print(num_columns)
                for col in num_columns:
                    stats = StatisticSimilarity.compute(
                        real_data=orig_data[col],
                        synthetic_data=privsmote_data[col],
                        statistic='mean'
                    )
                    range = RangeCoverage.compute(
                        real_data=orig_data[col],
                        synthetic_data=privsmote_data[col]
                    )
                    # print(stats)
                    stats_ = {'ds':privsmote, 'col': col, 'Statistic similarity': stats, 'Range Coverage': range, 'Category Coverage': np.nan}
                    all_stats.append(stats_)

                cat_columns = orig_data.select_dtypes(include=object).columns
                for col in cat_columns:
                    cat_range = CategoryCoverage.compute(
                        real_data=orig_data[col],
                        synthetic_data=privsmote_data[col]
                    )
                    stats_ = {'ds':privsmote, 'col': col, 'Statistic similarity': stats, 'Range Coverage': range, 'Category Coverage': cat_range}

                    all_stats.append(stats_)
                
df = pd.DataFrame(all_stats)
df.to_csv('output/coverage.csv', index=False)

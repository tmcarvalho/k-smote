## $\epsilon$-PrivateSMOTE

This project includes all the experimental evaluation of the $\epsilon$-PrivateSMOTE proposal.

### Data
All data used including the original files and all transformed data variants.

- [Original](https://www.kaggle.com/datasets/up201204722/original)
- [Transformed data](https://www.kaggle.com/datasets/up201204722/ppt-on-original-data) with ARX tool.
- [Data variants](https://www.kaggle.com/datasets/up201204722/privatesmote-data) generated with deep learning, differentially private-based solutions and $\epsilon$-PrivateSMOTE.


### Files description

**code/tranformations** Apply the transformations to the original data.

- task.py - puts in the queue all the original files to synthetise.

    Eg. ```python3 code/transformations/task_transf.py  --input_folder "original" --type "PrivateSMOTE"```
- main.py - apply transformation *type*: differentially private solutions (with Synthcity), deep learning solutions (with SDV) and $\epsilon$-PrivateSMOTE. It also includes the computational costs for generating each new data variant.

    Eg. ```python3 code/transformations/main.py --type "PrivateSMOTE"```


**code/modeling** Machine Learning pipeline for predictive performance evaluation.

- task.py - puts in the queue all the files for modeling.

    Eg. ```python3 code/modeling/task.py  --input_folder "output/oversampled/PrivateSMOTE"```
- worker.py - calls apply_models (data handler) to run all the Machine Learning models.

    Eg. ```python3 code/modeling/worker.py --type "PrivateSMOTE" --input_folder "output/oversampled/PrivateSMOTE" --output_folder "output/modeling/PrivateSMOTE"```

**code/record_linkage** Linkability for privacy risk evaluation.

- task_anonymeter.py - puts in the queue all the files for linkability analysis.

    Eg. ```python3 code/record_linkage/task_anonymeter.py  --input_folder  "output/oversampled/PrivateSMOTE"```
- worker_anonymeter.py - calls anonymeter.

    Eg. ```python3 code/record_linkage/worker_anonymeter.py --type "PrivateSMOTE" --input_folder "output/oversampled/PrivateSMOTE" --output_folder "output/anonymeter/PrivateSMOTE"```


**code**
- coverage.py - perform multiple generic data utility measures from SDVMetrics. 

    Eg. ```python3 code/coverage.py --input_folder "PrivateSMOTE"```

    **Analysis in notebooks**
- compt_costs.py - analysis of the computational costs in generating all data variants.
- results_coverage.py - analysis of the data utility measures for all data variants.
- results_performance.py - analysis of the predictive performance for all data variants.
- results_anonymeter.py - analysis of the linkability risk for all data variants.
- tradeoff.py - analysis of the tradeoff between predictive performance and linkability risk.
- bayesTest.py - analysis of the Bayesian Test for the oracle (best solutions in out of sample).


indexes.npy - indexes of the testing sets.

list_key_vars.csv - 5 random sets of quasi-identifiers for each data set.


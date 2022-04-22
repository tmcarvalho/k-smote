"""Record linkage
This script applies comparisons between the select QIs for all single outs.
"""
import recordlinkage
import pandas as pd

def record_linkage(transformed: pd.DataFrame,
    original: pd.DataFrame, columns: list) -> pd.DataFrame:
    """_summary_

    Args:
        transformed (pd.DataFrame): transformed dataframe with singleouts
        original (pd.DataFrame): original dataframe with single outs
        columns (list): list of quasi-identifiers

    Returns:
        pd.Dataframe: comparison results with respective score
    """
    indexer = recordlinkage.Index()
    indexer.full()
    candidates = indexer.index(transformed, original)
    print(len(candidates))
    compare = recordlinkage.Compare()
    for idx, col in enumerate(columns):
        compare.numeric(col, columns[idx], label=columns[idx], method='gauss')

    comparisons = compare.compute(candidates, transformed, original)
    potential_matches = comparisons[comparisons.sum(axis=1) > 1].reset_index()
    potential_matches['Score'] = potential_matches.iloc[:, 2:].sum(axis=1)

    return potential_matches

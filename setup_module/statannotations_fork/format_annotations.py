# %%
import os  # type:ignore # isort:skip # fmt:skip # noqa # nopep8
import sys  # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

mod = sys.modules[__name__]

code_dir = None
code_dir_name = 'Code'
unwanted_subdir_name = 'Analysis'

if code_dir_name not in str(Path.cwd()).split('/')[-1]:
    for _ in range(5):

        parent_path = str(Path.cwd().parents[_]).split('/')[-1]

        if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):

            code_dir = str(Path.cwd().parents[_])

            if code_dir is not None:
                break
else:
    code_dir = str(Path.cwd())
sys.path.append(code_dir)
if 'setup_module' not in sys.path:
    sys.path.append(f'{code_dir}/setup_module')
sys.path = list(set(sys.path))

# %load_ext autoreload
# %autoreload 2

# %%
from operator import itemgetter
from typing import List

import numpy as np
import pandas as pd
from statannotations_fork.stats.StatResult import StatResult


def pval_annotation_text(result: List[StatResult],
                         pvalue_thresholds: List) -> List[tuple]:
    """

    :param result: StatResult instance or list thereof
    :param pvalue_thresholds: thresholds to use for formatter
    :returns: A List of rendered annotations if a list of StatResults was
        provided, a string otherwise.
    """

    x1_pval = np.array([res.pvalue for res in result])

    pvalue_thresholds = sorted(pvalue_thresholds, key=itemgetter(0),
                               reverse=True)

    x_annot = pd.Series(['' for _ in x1_pval])

    for i, threshold in enumerate(pvalue_thresholds[:-1]):
        condition = ((x1_pval <= threshold[0])
                     & (pvalue_thresholds[i + 1][0] < x1_pval))
        x_annot[condition] = threshold[1]

    x_annot[x1_pval <= pvalue_thresholds[-1][0]] = pvalue_thresholds[-1][1]

    return [(star, res) for star, res in zip(x_annot, result)]


def simple_text(result: StatResult, pvalue_format, pvalue_thresholds,
                short_test_name=True) -> str:
    """
    Generates simple text for test name and pvalue.

    :param result: StatResult instance
    :param pvalue_format: format string for pvalue
    :param pvalue_thresholds: String to display per pvalue range
    :param short_test_name: whether to display the test (short) name
    :returns: simple annotation

    """
    # Sort thresholds
    thresholds = sorted(pvalue_thresholds, key=lambda x: x[0])

    text = (f'{result.test_short_name} '
            if short_test_name and result.test_short_name
            else '')

    for threshold in thresholds:
        if result.pvalue < threshold[0]:
            pval_text = 'p ≤ {}'.format(threshold[1])
            break
    else:
        pval_text = 'p = {}'.format(pvalue_format).format(result.pvalue)

    return result.adjust(text + pval_text)

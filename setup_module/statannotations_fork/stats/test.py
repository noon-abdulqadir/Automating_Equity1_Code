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
from typing import Union

import numpy as np
from statannotations_fork.stats.ComparisonsCorrection import (
    ComparisonsCorrection,
    get_validated_comparisons_correction,
)
from statannotations_fork.stats.StatResult import StatResult
from statannotations_fork.stats.StatTest import StatTest

IMPLEMENTED_TESTS = ['t-test_ind', 't-test_welch', 't-test_paired',
                     'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                     'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']


def apply_test(
        group_data1,
        group_data2,
        test: Union[StatTest, str] = None,
        comparisons_correction: Union[ComparisonsCorrection, str] = None,
        num_comparisons: int = 1,
        alpha: float = 0.05,
        **stats_params
):
    """Get formatted result of two-sample statistical test.

    :param group_data1: data
    :param group_data2: data
    :param test: Union[StatTest, str]: Statistical test to run.
        Either a `StatTest` instance or one of:
        - `Brunner-Munzel`
        - `Levene`
        - `Mann-Whitney`
        - `Mann-Whitney-gt`
        - `Mann-Whitney-ls`
        - `t-test_ind`
        - `t-test_welch`
        - `t-test_paired`
        - `Wilcoxon`
        - `Kruskal`

    :param comparisons_correction: Union[ComparisonsCorrection, str]:
        (Default value = None)
        Method to use for multiple comparisons correction. Either a
        `ComparisonsCorrection` instance or one of (interfacing statsmodels):
        - Bonferroni
        - Holm-Bonferroni
        - Benjamini-Hochberg
        - Benjamini-Yekutieli

    :param num_comparisons: int:  (Default value = 1)
        Number of comparisons to use for multiple comparisons correction.
    :param alpha: float:  (Default value = 0.05)
        Used for pvalue interpretation in case of comparisons_correction.
    :param stats_params: Additional keyword arguments to pass to the test
        function
    """
    # Check arguments.
    if (isinstance(comparisons_correction, ComparisonsCorrection)
            or comparisons_correction is None):
        pass
    else:
        comparisons_correction = \
            get_validated_comparisons_correction(comparisons_correction)

    if test is None:
        result = StatResult(None, '', None, None, np.nan)

    else:
        if isinstance(test, StatTest):
            get_stat_result = test
        else:
            get_stat_result = StatTest.from_library(test)

        result = get_stat_result(
            group_data1, group_data2, alpha=alpha, **stats_params)

    # Optionally, run multiple comparisons correction that can independently be
    # applied to each pval
    if comparisons_correction is not None and comparisons_correction.type == 0:
        result.pvalue = comparisons_correction(result.pvalue, num_comparisons)
        result.correction_method = comparisons_correction.name

    return result

# %%
import os # type:ignore # isort:skip # fmt:skip # noqa # nopep8
import sys # type:ignore # isort:skip # fmt:skip # noqa # nopep8
from pathlib import Path # type:ignore # isort:skip # fmt:skip # noqa # nopep8

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

# %load_ext autoreload
# %autoreload 2


# %%
from setup_module.imports import *  # type:ignore # isort:skip # fmt:skip # noqa # nopep8

# %%
# Compute Hotelling statistic
def hotelling(beta_IV, vcov_IV, model_unbias):
    b_diff = beta_IV - model_unbias.params
    var_diff = vcov_IV + model_unbias.cov_params()
    return float(np.dot(b_diff, np.dot(np.linalg.inv(var_diff), b_diff)))

# %%
# Compute correlations for diagnostics
def get_corrs(lhs, rhs):
    return np.abs(np.corrcoef(lhs.values, rhs.values.transpose()).mean())

# %%
# make formula for IV regression
def make_formula_endog_exog_instrument(regressor, control, IVs, var, type, data):
    regressor_ = regressor.replace('%', '').replace(' ', '_')
    control_ = [c.replace('%', '').replace(' ', '_') for c in control]
    IVs_ = [i.replace('%', '').replace(' ', '_') for i in IVs]
    var_ = var.replace('%', '').replace(' ', '_')

    if control:
        if type == 'XZ':
            formula_str = f'{regressor_} ~ {" + ".join(IVs_)}'
            endog_names = regressor
            exog_names = IVs
            instrument_names = None
        elif type == 'YX':
            formula_str = f'{var_} ~ {regressor_} + {" + ".join(control_)}'
            endog_names = var
            exog_names = [regressor] + control
            instrument_names = None
        elif type == 'all':
            formula_str = f'{var_} ~ {regressor_} + {" + ".join(control_)} | {" + ".join(IVs_)} + {" + ".join(control_)}'
            endog_names = var
            exog_names = [regressor] + control
            instrument_names = IVs + control
    elif type == 'XZ':
        formula_str = f'{regressor_} ~ {" + ".join(IVs_)}'
        endog_names = regressor
        exog_names = IVs
        instrument_names = None
    elif type == 'YX':
        formula_str = f'{var_} ~ {regressor_}'
        endog_names = var
        exog_names = regressor
    elif type == 'all':
        formula_str = f'{var_} ~ {regressor_} | {" + ".join(IVs_)}'
        endog_names = var
        exog_names = regressor
        instrument_names = IVs

    endog = data[endog_names]
    exog = data[exog_names]
    instrument = data[instrument_names]
    constant = sm.add_constant(exog)

    formula_data = data.copy()
    formula_data.columns = formula_data.columns.str.replace('%', '').str.replace(' ', '_')

    try:
        ols_model = smf.ols(formula=formula_str, data=formula_data)
    except:
        ols_model = sm.OLS(endog=endog, exog=exog, data=data)

    return formula_data, formula_str, ols_model, endog_names, endog, exog_names, exog, instrument_names, instrument, constant

# %%
#### Functions to select strong and valid IVs based on Lasso regression for ForestIV approach ####

# Use Lasso method to select strong IVs
# data_unlabel: unlabeled dataset
# regressor: name of the endogenous tree
# candidates: candidate IVs as a character vector of variable names
# Function to select strong IVs using Lasso
def lasso_select_strong(data_unlabel, regressor, candidates):
    formula_data = data_unlabel.copy()
    formula_data.columns = formula_data.columns.str.replace('%', '').str.replace(' ', '_')
    if len(candidates) != 0:
        formula_str = f'{regressor.replace("%", "").replace(" ", "_")} ~ {" + ".join([c.replace("%", "").replace(" ", "_") for c in candidates])}'
        y = formula_data[regressor]
        X = formula_data[candidates]

        lasso = LassoCV(cv=5)
        lasso.fit(X, y)
        selection = lasso.coef_ != 0
        return np.array(candidates)[selection]
    else:
        return candidates

# %%
# Use Lasso method to select valid IVs
# data_test: the testing dataset
# regressor: name of the endogenous tree
# candidates: candidate IVs as a character vector of variable names
# Function to select valid IVs using Lasso
def lasso_select_valid(col, data_test, regressor, candidates):
    if len(data_test) == 0 or len(candidates) == 0:
        return candidates
    focal_pred = data_test[regressor]
    others_pred = data_test[candidates]
    actual = data_test[f'{col}_actual']
    focal_error = focal_pred - actual

    lasso = LassoCV(cv=5)
    lasso.fit(others_pred, focal_error)
    invalid = lasso.coef_ == 0
    return np.array(candidates)[~invalid]

# %%
# Perform Lasso select for validity and strength for a given endogenous covariate
# data_test: the testing dataset
# data_unlabel: unlabeled dataset
# iterative: iterate between IV validity and strength selection? Default to TRUE
# ntree: number of trees in random forest
# regressor: name of the endogenous tree
# Function to perform Lasso selection for validity and strength
def lasso_select(col, data_test, data_unlabel, ntree, regressor, iterative):
    candidates = [f'{col}_tree_{i}' for i in range(0, ntree) if f'{col}_tree_{i}' != regressor]

    def get_corrs(lhs, rhs):
        return np.abs(np.corrcoef(lhs.values, rhs.values.transpose()).mean())

    pp_abs_before = get_corrs(data_unlabel[regressor], data_unlabel[candidates])
    pe_abs_before = get_corrs((data_test[regressor] - data_test[f'{col}_actual']), data_test[candidates])

    if iterative:
        IV_valid = lasso_select_valid(col, data_test, regressor, candidates)
        IVs = lasso_select_strong(data_unlabel, regressor, IV_valid)
        while len(IVs) != len(candidates):
            candidates = IVs
            IV_valid = lasso_select_valid(col, data_test, regressor, candidates)
            IVs = lasso_select_strong(data_unlabel, regressor, IV_valid)
    else:
        IV_valid = lasso_select_valid(col, data_test, regressor, candidates)
        IVs = lasso_select_strong(data_unlabel, regressor, IV_valid)

    if len(IVs) != 0:
        pp_abs_after = get_corrs(data_unlabel[regressor], data_unlabel[IVs])
        pe_abs_after = get_corrs(data_test[regressor] - data_test[f'{col}_actual'], data_test[IVs])
    else:
        pp_abs_after = np.nan
        pe_abs_after = np.nan

    return {
        "IVs": IVs,
        "correlations": [pp_abs_before, pe_abs_before, pp_abs_after, pe_abs_after]
    }

# %%
# Function to perform 2SLS estimation
def perform_2sls_estimation(data_unlabel_new, regressor, var, control, IVs, family):
    if family.__class__.__name__ == 'Gaussian' and family.link.__class__.__name__ == 'Identity':
        (
            formula_data, formula_str, ols_model, endog_names, endog, exog_names, exog, instrument_names, instrument, constant
        ) = make_formula(
            regressor, controls, IVs, var, 'all', data_unlabel_new
        )
        model_IV = IV2SLS(endog=endog, exog=constant, instrument=instrument).fit()
    else:
        print('Only Gaussian family implemented.')
    return model_IV

# %%
#' ForestIV Main Function
#'
#' This function implements the main ForestIV approach.
#'
#' @param data_test Testing dataframe for random forest, must have a column named "actual" that contains the ground truth, and all trees' predictions.
#' @param data_unlabel Unlabel dataframe for random forest, must have all trees' predictions.
#' @param control A character vector of control variable names. Pass an empty vector if there are no control variables
#' @param method "Lasso" for ForestIV method and "IIV" for EnsembleIV method.
#' @param iterative Whether to perform iterative IV selection or not, default to TRUE. Only relevant when method = "Lasso"
#' @param ntree Number of trees in the random forest.
#' @param model_unbias Unbiased estimation.
#' @param family Model specification, same as in the family parameter in glm.
#' @param diagnostic Whether to output diagnostic correlations for instrument validity and strength, default to TRUE.
#' @param select_method method of IV selection. One of "optimal" (LASSO based), "top3", and "PCA".
#' @return ForestIV estimation results
#' @export

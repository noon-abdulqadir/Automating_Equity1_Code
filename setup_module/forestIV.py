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
def hotelling(col, beta_IV, vcov_IV, model_unbias):
    index_name = beta_IV.index[beta_IV.index.str.startswith(f'{col}')][0]
    model_unbias.params = model_unbias.params.rename(
        index=lambda x: x
        if not x.startswith(col) or x == index_name
        else re.sub(re.escape(col) + r'\w*', index_name, x)
    )
    model_unbias_cov_params = model_unbias.cov_params().rename(
        index=lambda x: x
        if not x.startswith(col) or x == index_name
        else re.sub(re.escape(col) + r'\w*', index_name, x),
        columns=lambda x: x
        if not x.startswith(col) or x == index_name
        else re.sub(re.escape(col) + r'\w*', index_name, x)
    )
    b_diff = beta_IV - model_unbias.params
    var_diff = vcov_IV + model_unbias_cov_params
    return float(np.dot(b_diff, np.dot(np.linalg.inv(var_diff), b_diff)))

# %%
# Compute correlations for diagnostics
def get_corrs(lhs, rhs):
    return np.abs(np.corrcoef(lhs.values, rhs.values.transpose()).mean())

# %%
# make formula for IV regression
def make_formula_endog_exog_instrument(regressor, control, IVs, var, type, data):
    replace_arg = lambda x: x.replace('%', 'percentage').replace(' ', '_')
    regressor_ = replace_arg(regressor)
    if len(control) > 0:
        if isinstance(control, str):
            control_ = replace_arg(control)
        else:
            control_ = " + ".join([replace_arg(c) for c in control])
    if isinstance(IVs, str):
        IVs_ = replace_arg(IVs)
    else:
        IVs_ = " + ".join([replace_arg(i) for i in IVs])
    if isinstance(var, str):
        var_ = replace_arg(var)
    else:
        var_ = " + ".join([replace_arg(v) for v in var])

    if len(control) > 0:
        if type == 'XZ':
            formula_str = f'{regressor_} ~ {IVs_}'
            endog_names = regressor
            exog_names = IVs
            instrument_names = None
        elif type == 'YX':
            formula_str = f'{var_} ~ {regressor_} + {control_}'
            endog_names = var
            exog_names = [regressor] + control
            instrument_names = None
        elif type == 'all':
            formula_str = f'{var_} ~ {regressor_} + {control_} | {IVs_} + {control_}'
            endog_names = var
            exog_names = [regressor] + control
            instrument_names = IVs + control
    elif type == 'XZ':
        formula_str = f'{regressor_} ~ {IVs_}'
        endog_names = regressor
        exog_names = IVs
        instrument_names = None
    elif type == 'YX':
        formula_str = f'{var_} ~ {regressor_}'
        endog_names = var
        exog_names = regressor
    elif type == 'all':
        formula_str = f'{var_} ~ {regressor_} | {IVs_}'
        endog_names = var
        exog_names = regressor
        instrument_names = IVs

    endog = data[endog_names]
    exog = data[exog_names]
    instrument = data[instrument_names]

    formula_data = data.copy()
    formula_data = formula_data.rename(columns=replace_arg)

    try:
        ols_model = smf.ols(formula=formula_str, data=formula_data)
    except:
        ols_model = sm.OLS(endog=endog, exog=exog, data=data)

    return formula_data, formula_str, ols_model, endog_names, endog, exog_names, exog, instrument_names, instrument

# %%
#### Functions to create valid IVs from imperfect IVs for EnsembleIV approach ####
# Based on imperfect IV work: Nevo and Rosen (2012); Clarke and Matt (2017)


# Create valid IVs from imperfect ones using testing dataset, then transform IVs in unlabeled dataset
# data_test: the testing dataset
# regressor: name of the endogenous tree
# candidates: candidate IVs as a character vector of variable names
def IIVCreate_Valid(col, data_test, data_unlabel, regressor, candidates):
    if len(data_test) == 0 or len(candidates) == 0:
        return candidates

    data_unlabel_new = data_unlabel.copy()
    data_test_new = data_test.copy()

    # Perform IV creation and transformation
    x_unlabel = data_unlabel[regressor]
    x_test = data_test[regressor]
    focal_error = x_test - data_test[f'{col}_actual']
    sigma_x = np.std(x_test)
    cov_xe = np.cov(x_test, focal_error)[0, 1]

    for candidate in candidates:
        z_test = data_test[candidate]
        z_unlabel = data_unlabel[candidate]
        sigma_z = np.std(z_test)
        cov_ze = np.cov(z_test, focal_error)[0, 1]

        lambda_ = (cov_ze / cov_xe) * (sigma_x / sigma_z)

        # Transform the corresponding IV in the unlabeled dataset
        # Also transform the testing dataset (for diagnostic purposes only)
        data_unlabel_new[candidate] = sigma_x * z_unlabel - lambda_ * sigma_z * x_unlabel
        data_test_new[candidate] = sigma_x * z_test - lambda_ * sigma_z * x_test

    return {'data_unlabel_new': data_unlabel_new, 'data_test_new': data_test_new}

# %%
# Use Lasso method to select strong IVs, only to be used after IIVCreate_Valid
# data_unlabel_new: the transformed unlabeled dataset, output of IIVCreate_Valid function
# regressor: name of the endogenous tree
# candidates: candidate IVs as a character vector of variable names
def IIVSelect_Strong(data_unlabel_new, regressor, candidates):
    replace_arg = lambda x: x.replace('%', 'percentage').replace(' ', '_')
    # Create a formula for the regression model
    formula_data = data_unlabel.copy()
    formula_data = formula_data.rename(columns=replace_arg)
    formula_str = f'{replace_arg(regressor)} ~ {" + ".join([replace_arg(c) for c in candidates])}'
    formula = sm.formula.ols(formula_str, data=formula_data)

    # Fit a Lasso regression model
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(formula_data[[replace_arg(c) for c in candidates]], formula_data[replace_arg(regressor)])

    return [
        [replace_arg(c) for c in candidates][i]
        for i, coef in enumerate(lasso_model.coef_)
        if coef != 0
    ]

# %%
# Perform IIVCreate to get valid IVs, then use Lasso to select strong ones
# data_test: the testing dataset
# data_unlabel: unlabeled dataset
# ntree: number of trees in random forest
# regressor: name of the endogenous tree
# select_method: method of IV selection. Only for research purpose, will remove in production version
def IIVSelect(col, data_test, data_unlabel, ntree, regressor, select_method):
    candidates = [f'{col}_tree_{i}' for i in range(0, ntree) if f'{col}_tree_{i}' != regressor]

    pp_abs_before = get_corrs(data_unlabel[regressor], data_unlabel[candidates])
    pe_abs_before = get_corrs(data_test[regressor] - data_test[f'{col}_actual'], data_test[candidates])

    data_unlabel_new = IIVCreate_Valid(col, data_test, data_unlabel, regressor, candidates)['data_unlabel_new']
    data_test_new = IIVCreate_Valid(col, data_test, data_unlabel, regressor, candidates)['data_test_new']

    if select_method == 'optimal':
        IVs = IIVSelect_Strong(data_unlabel_new, regressor, candidates)

    if select_method == 'top3':
        corrs = data_unlabel_new[candidates].corrwith(data_unlabel_new[regressor])
        IVs = list(corrs.abs().nlargest(3).index)

    if select_method == 'PCA':
        ncomp = 3
        IVs = [f'PCA_IV{i}' for i in range(1, ncomp + 1)]
        pca = PCA(n_components=ncomp)
        pca.fit(data_unlabel_new[candidates])
        data_unlabel_new[IVs] = pca.transform(data_unlabel_new[candidates])
        data_test_new[IVs] = pca.transform(data_test_new[candidates])

    if len(IVs) != 0:
        pp_abs_after = get_corrs(data_unlabel_new[regressor], data_unlabel_new[IVs])
        pe_abs_after = get_corrs(data_test_new[regressor] - data_test_new[f'{col}_actual'], data_test_new[IVs])
    else:
        pp_abs_after = np.nan
        pe_abs_after = np.nan

    return {
        'IVs': IVs,
        'data_unlabel_new': data_unlabel_new,
        'correlations': [pp_abs_before, pe_abs_before, pp_abs_after, pe_abs_after]
    }

# %%
#### Functions to select strong and valid IVs based on Lasso regression for ForestIV approach ####

# Use Lasso method to select strong IVs
# data_unlabel: unlabeled dataset
# regressor: name of the endogenous tree
# candidates: candidate IVs as a character vector of variable names
# Function to select strong IVs using Lasso
def lasso_select_strong(data_unlabel, regressor, candidates):
    replace_arg = lambda x: x.replace('%', 'percentage').replace(' ', '_')
    formula_data = data_unlabel.copy()
    formula_data = formula_data.rename(columns=replace_arg)
    if len(candidates) != 0:
        formula_str = f'{replace_arg(regressor)} ~ {" + ".join([replace_arg(c) for c in candidates])}'
        y = formula_data[replace_arg(regressor)]
        X = formula_data[[replace_arg(c) for c in candidates]]

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

    pp_abs_before = get_corrs(data_unlabel[regressor], data_unlabel[candidates])
    pe_abs_before = get_corrs((data_test[regressor] - data_test[f'{col}_actual']), data_test[candidates])

    if iterative:
        IV_valid = lasso_select_valid(col, data_test, regressor, candidates)
        IVs = lasso_select_strong(data_unlabel, regressor, IV_valid)
        while len(IVs) != len(candidates):
            print('-'*20)
            print('Iterating...')
            print(f'{len(IVs)} IVs selected')
            print(f'{len(candidates) - len(IVs)} IVs remaining')
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
            formula_data, formula_str, ols_model, endog_names, endog, exog_names, exog, instrument_names, instrument
        ) = make_formula_endog_exog_instrument(
            regressor, control, IVs, var, 'all', data_unlabel_new
        )
        model_IV = IV2SLS(endog=endog, exog=exog, instrument=instrument)
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
# ForestIV Main Function (Python implementation)
def forest_iv(col, data_test, data_unlabel, var, ntree, model_unbias, control=None, family=None, select_method=None, method=None, iterative=None, diagnostic=None):
    """ForestIV Main Function
    This function implements the main ForestIV approach.

    Args:
        col: Name of classified variable
        data_test: Testing dataframe for random forest, must have a column named "{col}_actual" that contains the ground truth, and all trees' predictions. data_test = pd.DataFrame(test[f'{col}_indiv_pred_test'], test[f'{col}_aggr_pred_test'], f'test[f'{col}_actual])
        data_unlabel: Unlabel dataframe for random forest, must have all trees' predictions. data_unlabel = pd.DataFrame(df_unlabeled, df_unlabeled[f'{col}_indiv_pred_unlabel], df_unlabeled[f'{col}_aggr_pred_unlabel'])
        control: A character vector of control variable names. Pass an empty vector if there are no control variables
        method: "Lasso" for ForestIV method and "IIV" for EnsembleIV method.
        iterative: Whether to perform iterative IV selection or not, default to TRUE. Only relevant when method = "Lasso"
        ntree: Number of trees in the random forest.
        model_unbias: Unbiased estimation.
        family: Model specification, same as in the family parameter in glm.
        diagnostic: Whether to output diagnostic correlations for instrument validity and strength, default to TRUE.
        select_method: method of IV selection. One of "optimal" (LASSO based), "top3", and "PCA".
    Returns:
        ForestIV estimation results
    """

    if control is None:
        control = []
    if family is None:
        family = sm.families.Gaussian(link=sm.families.links.Identity())
    if select_method is None:
        select_method = 'optimal'
    if method is None:
        method = 'Lasso'
    if iterative is None:
        iterative = True
    if diagnostic is None:
        diagnostic = True
    results = []

    for i in tqdm.tqdm(range(0, ntree)):
        regressor = f'{col}_tree_{i}'
        print('-'*20)
        print(f'Analyzing {regressor}/{ntree-1} trees')

        if method == 'Lasso':
            output = lasso_select(col, data_test, data_unlabel, ntree, regressor, iterative)
            IVs = output['IVs'].tolist()
            data_unlabel_new = data_unlabel.copy()

        if method == 'IIV':
            output = iiv_select(col, data_test, data_unlabel, ntree, regressor, select_method)
            IVs = output['IVs'].tolist()
            data_unlabel_new = output['data_unlabel_new']

        if len(IVs) != 0:
            model_IV = perform_2sls_estimation(data_unlabel_new, regressor, var, control, IVs, family)
            results_IV = model_IV.fit()
            beta_IV = results_IV.params
            vcov_IV = results_IV.cov_params()
            se_IV = np.sqrt(np.diag(vcov_IV))
            convergence = 0
            H_stats = hotelling(col, beta_IV, vcov_IV, model_unbias)
            correlations = output['correlations']
            results.append([*beta_IV, *se_IV, H_stats, convergence, *correlations])

    results = pd.DataFrame(results, columns=[f'beta_{i}' for i in range(0, len(beta_IV))] +
                                            [f'se_{i}' for i in range(0, len(se_IV))] +
                                            ['Hotelling', 'Convergence', 'pp_abs_before', 'pe_abs_before', 'pp_abs_after', 'pe_abs_after'])
    if not diagnostic:
        results = results.iloc[:, :-4]

    return results_IV, output, results

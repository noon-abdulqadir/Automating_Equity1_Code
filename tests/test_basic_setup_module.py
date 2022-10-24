# -*- coding: utf-8 -*-
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Install packages and import
# %%
# #################################### PLEASE INSTALL LATEST CHROME WEBDRIVER #####################################
# Uncomment to run as required
# #     --install-option="--chromedriver-version= *.**" \
#   --install-option="--chromedriver-checksums=4fecc99b066cb1a346035bf022607104,058cd8b7b4b9688507701b5e648fd821"
# %%
# ##### COPY THE LINES IN THIS COMMENT TO THE TOP OF NEW SCRIPTS #####
# # Function to import this package to other files
# import io
# import os
# import sys
# from pathlib import Path
# main_dir_name = 'Code'
# unwanted_subdir_name = 'Analysis'
# for _ in range(5):
#     parent_path = str(Path.cwd().parents[_]).split('/')[-1]
#     if (main_dir_name in str(Path.cwd()).split('/')[-1]) and (unwanted_subdir_name not in str(Path.cwd()).split('/')[-1]):
#         code_dir = str(Path.cwd())
#     elif (main_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):
#         code_dir = str(Path.cwd().parents[_])
#     if code_dir:
#         break
# main_dir = str(Path(code_dir).parents[0])
# scraped_data = f'{code_dir}/scraped_data'
# sys.path.append(code_dir)
# from setup_module.imports import *
# from setup_module.params import *
# from setup_module.scraping import *
# from setup_module.classification import *
# from setup_module.vectorizers_classifiers import *
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib notebook
# %matplotlib inline
# %%
import io
import os
import sys
from pathlib import Path

main_dir_name = 'Code'
unwanted_subdir_name = 'Analysis'

for _ in range(5):
    parent_path = str(Path.cwd().parents[_]).split('/')[-1]
    if (main_dir_name in str(Path.cwd()).split('/')[-1]) and (
        unwanted_subdir_name not in str(Path.cwd()).split('/')[-1]
    ):
        code_dir = str(Path.cwd())

    elif (main_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):
        code_dir = str(Path.cwd().parents[_])

    if code_dir:
        break

main_dir = str(Path(code_dir).parents[0])
scraped_data = f'{code_dir}/scraped_data'
sys.path.append(code_dir)

from setup_module.classification import *
from setup_module.imports import *
from setup_module.params import *
from setup_module.scraping import *
from setup_module.vectorizers_classifiers import *

warnings.filterwarnings('ignore', category=DeprecationWarning)


# %%
class EstimatorSelectionHelper:

    def __init__(self, vectorizers_pipe, selectors_pipe, classifiers_pipe):
        self.search_dict = {}
        self.vectorizers_pipe = vectorizers_pipe
        self.selectors_pipe = selectors_pipe
        self.classifiers_pipe = classifiers_pipe

        # for vectorizer_name, vectorizer_and_params in self.vectorizers_pipe.items():
        #     self.vectorizer_name = vectorizer_name
        #     self.vectorizer = vectorizer_and_params[0]
        #     self.vectorizer_params = vectorizer_and_params[1]
        # for selector_name, selector_and_params in self.selectors_pipe.items():
        #     self.selector_name = selector_name
        #     self.selector = selector_and_params[0]
        #     self.selector_params = selector_and_params[1]
        # for classifier_name, classifier_and_params in classifiers_pipe.items():
        #     self.classifier_name =classifier_name
        #     self.classifier = classifier_and_params[0]
        #     self.classifier_params = classifier_and_params[1]

    def fit(self, X_train, y_train, col, search=RandomizedSearchCV, cv=cv, n_jobs=-1, verbose=3, scoring=scores, refit='recall', return_train_score=True,select_best_enabled=select_best_enabled,):
        for vectorizer_name, vectorizer_and_params in self.vectorizers_pipe.items():
            self.vectorizer_name = vectorizer_name
            self.vectorizer = vectorizer_and_params[0]
            self.vectorizer_params = vectorizer_and_params[1]
            for selector_name, selector_and_params in self.selectors_pipe.items():
                self.selector_name = selector_name
                self.selector = selector_and_params[0]
                self.selector_params = selector_and_params[1]
                for classifier_name, classifier_and_params in classifiers_pipe.items():
                    self.classifier_name =classifier_name
                    self.classifier = classifier_and_params[0]
                    self.classifier_params = classifier_and_params[1]

                    # Pipeline
                    if select_best_enabled is True:
                        ## Steps
                        self.steps = [
                            (self.vectorizer_name, self.vectorizer),
                            (self.selector_name, self.selector),
                            (self.classifier_name, self.classifier)
                        ]
                        ## Params
                        self.param_grid = {
                            **self.vectorizer_params,
                            **self.selector_params,
                            **self.classifier_params,
                        }
                        ## Pipeline
                        self.pipe = Pipeline(steps=self.steps)

                        ## Vectorizers, selectors, classifiers
                        self.vectorizer = self.pipe[:-2]
                        self.selector = self.pipe[:-1]
                        self.classifier = self.pipe[:]

                    elif select_best_enabled is False:
                        ## Steps
                        self.steps = [
                            (self.vectorizer_name, self.vectorizer),
                            (self.classifier_name, self.classifier)
                        ]
                        ## Params
                        self.param_grid = {
                            **self.vectorizer_params,
                            **self.classifier_params,
                        }
                        ## Pipeline
                        self.pipe = Pipeline(steps=steps)

                        ## Vectorizers, selectors, classifiers
                        self.vectorizer = self.pipe[:-1]
                        self.classifier = self.pipe[:]

                    print(f'Running {search.__name__} for {str(col)} - {self.vectorizer_name} + {self.classifier_name}')
                    searchcv = search(
                        estimator=self.pipe,
                        param_distributions=self.param_grid,
                        n_jobs=-1,
                        scoring=scores,
                        cv=cv,
                        refit=scores[0],
                        return_train_score=True,
                        verbose=3,
                    )
                    searchcv.fit(X_train, y_train)
                    self.search_dict[self.classifier_name] = searchcv

    # def score_summary(self, sort_by='mean_score'):
    #     def row(key, scores, params):
    #         d = {
    #                 'estimator': key,
    #                 'min_score': min(scores),
    #                 'max_score': max(scores),
    #                 'mean_score': np.mean(scores),
    #                 'std_score': np.std(scores),
    #         }
    #         return pd.Series({**params,**d})

    #     rows = []
    #     for k in self.search_dict:
    #         print(k)
    #         params = self.search_dict[k].cv_results_['params']
    #         scores = []
    #         for i in range(self.search_dict[k].cv):
    #             key = f'split{i}_test_score'
    #             r = self.search_dict[k].cv_results_[key]
    #             scores.append(r.reshape(len(params),1))

    #         all_scores = np.hstack(scores)
    #         for p, s in zip(params,all_scores):
    #             rows.append((row(k, s, p)))

    #     df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

    #     columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
    #     columns = columns + [c for c in df.columns if c not in columns]

    #     return df[columns]

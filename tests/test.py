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

from setup_module.imports import *
from setup_module.params import *
from setup_module.scraping import *
from setup_module.classification import *
from setup_module.vectorizers_classifiers import *

warnings.filterwarnings('ignore', category=DeprecationWarning)


class PostCleanUp:
    def __init__(
        self,
        site_from_list=True,
        site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
        keyword='',
        keywords_from_list=True,
        keywords_list=[],
        job_id_save_enabled=True,
        args=get_args(),
    ):
        self.site_from_list = site_from_list
        self.site_list = site_list
        self.keyword = keyword
        self.keywords_from_list = keywords_from_list
        self.keywords_list = keywords_list
        self.job_id_save_enabled = job_id_save_enabled
        self.args = args

    def get_site_from_list(self):

        if self.site_from_list  is True:
            self.glob_paths = [x for self.site in self.site_list for x in glob.glob(f'{scraped_data}/{str(self.site)}/Data/*.json')]
        elif self.site_from_list is False:
            self.glob_paths = glob.glob(f'{scraped_data}/*/Data/*.json')

        yield self.glob_paths

    def get_keyword_from_list(self):

        if self.keyword_from_list  is True:
            self.keyword = ', '.join(glob_path.split('dict_')[1].split('.json')[0] for glob_path in self.glob_paths)

        yield self.keyword

    def make_final(post_cleanup):
        def make_final_site(post_cleanup):

            self.df_list_from_site = []
            self.df_jobs = self.df_list_from_site.append(self.df_list_from_keyword)
            if self.args['save_enabled'] is True:
                with open(f'{self.args["df_dir"]}df_{self.site}_all_jobs.{file_save_format}', 'wb') as f:
                    pickle.dump(self.df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)

            return self.df_jobs

        def make_final_keyword(post_cleanup):

            self.df_list_from_keyword = []
            self.df_jobs = self.df_list_from_keyword.append(post_cleanup)
            if self.args['save_enabled'] is True:
                with open(self.args['df_dir'] + 'df_all_jobs.p', 'wb') as f:
                    pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)

            return self.df_jobs

        if site_from_list is True:
            return make_final_site(post_cleanup)
        if keywords_from_list is True:
            return make_final_keyword(post_cleanup)

    @make_final
    def post_cleanup(self):
        try:
            self.df_jobs = self.post_cleanup_helper(self.keyword, self.site)

            if ((self.df_jobs.empty) or (len(self.df_jobs == 0))) and self.args['print_enabled'] is True:
                print(f'DF {self.keyword.title()} not collected yet.')
        except Exception:
            if self.args['print_enabled'] is True:
                print(f'{keyword} Not found: {Exception}.')
            self.df_jobs = pd.DataFrame()

        return self.df_jobs


    def post_cleanup_helper(self, self.keyword, self.site):
        (
            keyword_url,
            keyword_file,
            save_path,
            json_file_name,
            df_file_name,
            logs_file_name,
            filemode,
        ) = main_info(self.keyword, self.site)

        jobs, df_old_jobs = load_merge_dict_df(
            self.keyword, save_path, df_file_name, json_file_name
        )

        if (os.path.isfile(save_path + df_file_name.lower())) or (
            os.path.isfile(save_path + json_file_name.lower())
        ):
            with open(save_path + json_file_name, 'w', encoding='utf8') as f:
                json.dump(jobs, f)
            self.df_jobs = pd.DataFrame(jobs)
            if (not self.df_jobs.empty) and (len(self.df_jobs != 0)):
                if args['print_enabled'] is True:
                    print(f'Cleaning and saving {self.keyword} df.')
                self.df_jobs = clean_df(self.df_jobs)
                # Save df as csv
                if args['save_enabled'] is True:
                    self.df_jobs = save_df(
                        self.keyword,
                        self.df_jobs,
                        save_path,
                        keyword_file.lower(),
                        df_file_name.lower(),
                    )
            elif (self.df_jobs.empty) or (len(self.df_jobs == 0)):
                if args['print_enabled'] is True:
                    print(
                        f'Jobs DataFrame is empty since no jobs results were found for {str(keyword)}.'
                    )

        elif (not os.path.isfile(save_path + df_file_name.lower())) or (
            not os.path.isfile(save_path + json_file_name.lower())
        ):
            if args['print_enabled'] is True:
                print(f'No jobs file found for {self.keyword} in path: {save_path}.')
            self.df_jobs = pd.DataFrame()

        return self.df_jobs


def post_cleanup(
    keyword='',
    keywords_from_list=True,
    keywords_list=[],
    site_from_list=True,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    job_id_save_enabled=True,
    args=get_args(),
):
    print(
        f'NOTE: The function "post_cleanup" contains the following optional (default) arguments:\n{get_default_args(post_cleanup)}'
    )
    print('-' * 20)

    if site_from_list is True:
        df_list_from_site = []
        for site in site_list:
            if args['print_enabled'] is True:
                print('-' * 20)
                print(f'Cleaning up LIST OF DFs for {site}.')
            if keywords_from_list is True:
                df_list_from_keyword = []
                glob_paths = glob.glob(f'{code_dir}/{str(site)}/Data/*.json')
                for glob_path in glob_paths:
                    keyword = glob_path.split('dict_')[1].split('.json')[0]
                    if args['print_enabled'] is True:
                        print(f'Post collection cleanup for {keyword}.')
                    try:
                        df_jobs = post_cleanup_helper(keyword, site)

                        if df_jobs.empty or (len(df_jobs == 0)):
                            if args['print_enabled'] is True:
                                print(f'DF {keyword.title()} not collected yet.')
                    except Exception:
                        if args['print_enabled'] is True:
                            print(f'An error occured with finding DF {keyword}.')
                        df_jobs = pd.DataFrame()
                    else:
                        if args['print_enabled'] is True:
                            print(f'Cleaning up LIST OF DFs for {keyword}.')
                        if (not df_jobs.empty) and (len(df_jobs != 0)):
                            df_list_from_keyword.append(df_jobs)
                        df_list_from_site.append(df_list_from_keyword)
                        df_jobs = df_list_from_site
                        # pbar.finish()
            elif keywords_from_list is False:
                if args['print_enabled'] is True:
                    print(f'Post collection cleanup for {keyword}.')
                df_jobs = post_cleanup_helper(keyword, site)

                if (not df_jobs.empty) and (len(df_jobs != 0)):
                    df_list_from_site.append(df_jobs)
                df_jobs = df_list_from_site
            if args['save_enabled'] is True:
                with open(f'{args["df_dir"]}df_{site}_all_jobs.{file_save_format}', 'wb') as f:
                    pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
    #         pbar.finish()

    elif site_from_list is False:
        if keywords_from_list is True:
            df_list_from_keyword = []
            glob_paths = glob.glob(f'{scraped_data}/*/Data/*.json')
            for glob_path in glob_paths:
                keyword = glob_path.split('dict_')[1].split('.json')[0]
                if args['print_enabled'] is True:
                    print(f'Post collection cleanup for {keyword}.')
                try:
                    df_jobs = post_cleanup_helper(keyword, site)
                    if (df_jobs.empty) or (len(df_jobs == 0)):
                        if args['print_enabled'] is True:
                            print(f'DF {keyword.title()} not collected yet.')
                except Exception:
                    if args['print_enabled'] is True:
                        print(f'An error occured with finding DF {keyword}.')
                    df_jobs = pd.DataFrame()
                else:
                    if args['print_enabled'] is True:
                        print(f'Cleaning up LIST OF DFs for {keyword}.')
                    if (not df_jobs.empty) and (len(df_jobs != 0)):
                        df_list_from_keyword.append(df_jobs)
                    df_jobs = df_list_from_keyword
        #             pbar.finish()
        elif keywords_from_list is False:
            if args['print_enabled'] is True:
                print(f'Post collection cleanup for {keyword}.')
            try:
                df_jobs = post_cleanup_helper(keyword, site)

                if (df_jobs.empty) or (len(df_jobs == 0)):
                    if args['print_enabled'] is True:
                        print(f'DF {keyword.title()} not collected yet.')
            except Exception:
                if args['print_enabled'] is True:
                    print(f'An error occured with finding DF {keyword}.')
                df_jobs = pd.DataFrame()
            else:
                if args['print_enabled'] is True:
                    print(f'Cleaning up DF {keyword}.')

    if args['save_enabled'] is True:
        with open(args['df_dir'] + f'df_all_jobs.p', 'wb') as f:
            pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)

    if job_id_save_enabled is True:
        job_id_dict = make_job_id_v_genage_key_dict(site_from_list, site_list)
        sector_vs_job_id_dict = make_job_id_v_sector_key_dict()

    return df_jobs

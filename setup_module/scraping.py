# -*- coding: utf-8 -*-
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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


def get_main_path(
    code_dir=None,
    code_dir_name='Code',
    unwanted_subdir_name='Analysis',
    errors=(
        TypeError,
        AttributeError,
        ElementClickInterceptedException,
        ElementNotInteractableException,
        NoSuchElementException,
        NoAlertPresentException,
        TimeoutException,
    )
):

    print(
        f'NOTE: The function "get_main_path" contains the following optional (default) arguments:\nmain_dir_name: {main_dir_name} unwanted_subdir_name: {unwanted_subdir_name}'
    )
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


    return code_dir, errors


# %%

# %% [markdown]
# driver_browser_window

# %%
# Function to to check file exists and not empty
def is_non_zero_file(fpath):

    # check = (os.path.isfile(fpath) and os.path.getsize(fpath) > 0)
    # if check is True:
    #     print(f'File {fpath.split("/")[-1]} exists.')
    # elif check is False:
    #     print(f'File {fpath.split("/")[-1]} does not exist.')
    return (os.path.isfile(fpath) and os.path.getsize(fpath) > 0)


# %%
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# %%
# Function to validate path or file
def validate_path(file: str, file_extensions=None) -> str:

    if file_extensions is None:
        file_extensions = ['.*', 'chromedriver']
    if file.endswith(tuple(file_extensions)):
        if not os.path.isdir(file):
            if is_non_zero_file(file) is False:
                # file = input(f'No file found at {file}.\nPlease enter correct path.')
                try:
                    print(f'File {file} not found.')
                except Exception as e:
                    print(e.json())
    elif not file.endswith(tuple(file_extensions)):
        if not os.path.isdir(file):
            if is_non_zero_file(file) is False:
                # file = input(f'No file found at {file}.\nPlease enter correct path.')
                try:
                    print(f'File {file} not found.')
                except Exception as e:
                    print(e.json())

    return file


# %%
# Function to assign a value to multiple variables
def assign_all(number: int, value):
    return [value] * number


# nones = lambda n: [None for _ in range(n)]


# %%
# Function to check if list int values are increasing (monotonically)
def pairwise(seq):
    items = iter(seq)
    last = next(items)
    for item in items:
        yield last, item
        last = item


# %%
# Function to flatten nested items
def recurse(lst: list, function):
    for x in lst:
        try:
            yield func(x)
        except Exception:
            continue


# %%
# Function to print keys and values of nested dicts
def recursive_items(dictionary: dict, return_value_enabled: bool = False):
    for key, value in dictionary.items():
        yield (key)
        if return_value_enabled is True:
            yield (value)
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key)
            if return_value_enabled is True:
                yield (value)


# %%
# Function to flatten nested json dicts
def flatten(x: dict) -> dict:
    d = copy.deepcopy(x)
    for key in list(d):
        if isinstance(d[key], list):
            value = d.pop(key)
            for i, v in enumerate(value):
                d.update(flatten({f'{key}_{i}': v}))
        elif isinstance(d[key], dict):
            d[key] = str(d[key])
    return d


# %%
# Remove duplicates from dict
def remove_dupe_dicts(l, jobs=None, seen_ad=None, seen_id=None):
    if jobs is None:
        jobs = []
    if seen_ad is None:
        seen_ad = set()
    if seen_id is None:
        seen_id = set()

    for n, i in enumerate(l):
        if i not in l[n + 1:] and l['Job Description'] not in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan'] and len(l['Job Description']) != 0:
            jobs.append(i)

    return jobs


# %%
# Function to retry
def retry_on(exceptions, times, sleep_sec=1):

    def decorator(func):
        @wraps(func)
        def wrapper(args, **kwargs):
            last_exception = None
            for _ in range(times):
                try:
                    return func(args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not isinstance(last_exception, exceptions):
                        raise  # re-raises unexpected exceptions
                    sleep(sleep_sec)
            raise last_exception  # re-raises if attempts are unsuccessful

        return wrapper

    return decorator


# %%
# Decorator to retry
def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):

    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = f'{str(e)}, Retrying in {mdelay} seconds...'
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


# How to use
# @retry(Exception, tries=4)
# def test_fail(text):
#     raise Exception("Fail")

# %%
# Function to get function default values
def get_default_args(func) -> dict:

    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if (v.default is not inspect.Parameter.empty) and (k != 'args')
    }


# %%
def perfect_eval(anonstring):
    try:
        ev = ast.literal_eval(anonstring)
        return ev
    except ValueError:
        corrected = "\'" + anonstring + "\'"
        ev = ast.literal_eval(corrected)

        return ev


# %%
# Function to check internet connection
def is_connected(driver) -> bool:
    try:
        socket.create_connection(('1.1.1.1', 53))
        return True
    except OSError:
        print(
            'Internet is NOT connected. Sleeping for 10 seconds. Please check connection.'
        )
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#HeroSearchButton'))
        )
        is_connected(driver)

    return False


# %%
# Get driver path and select
def select_driver(webdriver_path=f'{code_dir}/setup_module/WebDriver/') -> str:

    plat = str(platform.system())
    py_version = float(sys.version[:3])
    if py_version < 2.0:
        print('Please update your python to at least version 3.')
    if plat == 'Darwin':
        DRIVER_PATH = validate_path(
            f'{webdriver_path}macOS_chromedriver'
        )
    elif plat == 'Windows':
        DRIVER_PATH = validate_path(
            f'{webdriver_path}win32_chromedriver'
        )
    elif plat == 'Linux':
        DRIVER_PATH = validate_path(
            f'{webdriver_path}linux64_chromedriver'
        )
    else:
        print(
            f'Cannot identify current platform.\nATTN: !!! Please download appropriate chrome driver and place inside the folder called "Webdriver" inside path: {current_file_path_parent}.'
        )
        driver_name = input(
            'Write the name of driver you place inside the Webdriver folder.'
        )
        DRIVER_PATH = validate_path(
            f'{webdriver_path}{driver_name}'
        )

    return DRIVER_PATH


# %%
# Get driver and set up parameters
def get_driver(
    select_driver,
    incognito_enabled: bool = True,
    headless_enabled: bool = False,
    proxy_enabled: bool = False,
):

    print(
        f'NOTE: The function "get_driver" contains the following optional (default) arguments:\n{get_default_args(get_driver)}'
    )
    print(f'Current path to chromewebdriver: {select_driver()}.')

    # Caps
    caps = DesiredCapabilities.CHROME
    caps['loggingPrefs'] = {
        'browser': 'WARNING',
        'driver': 'WARNING',
        'performance': 'WARNING',
    }
    caps['acceptSslCerts'] = True
    # Options
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('allow-elevated-browser')
    options.add_argument('window-size=1500,1200')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)

    if incognito_enabled is True:
        print('Incognito mode enabled.')
        options.add_argument('--incognito')
    elif incognito_enabled is False:
        print('No incognito mode enabled.')

    if headless_enabled is True:
        print('Headless driver enabled.')
        # options.add_argument('--headless')
        options.headless = True
    elif headless_enabled is False:
        print('No headless driver enabled.')
        options.headless = False

    # # Proxy
    # if proxy_enabled is True:
    #     print('Proxy browsing enabled.')
    #     ua = UserAgent()
    #     userAgent = ua.random
    #     req_proxy = RequestProxy()
    #     proxies = req_proxy.get_proxy_list()
    #     PROXY = proxies[5].get_address()
    #     print(f'Proxy country: {proxies[5].country}')
    #     caps['proxy'] = {
    #         'httpProxy': PROXY,
    #         'ftpProxy': PROXY,
    #         'sslProxy': PROXY,
    #         'proxyType': 'MANUAL',
    #     }
    #     options.add_argument(f'--proxy-server= {PROXY}')
    #     options.add_argument(f'user-agent={userAgent}')
    # elif proxy_enabled is False:
    #     print('No proxy browsing enabled.')
    #     pass

    driver = webdriver.Chrome(
        options=options,
        desired_capabilities=caps,
        # service_args=[f'--verbose", "--log-path={MyWriter.LOGS_PATH}'],
    )
    # http = urllib3.PoolManager(num_pools=500)
    warnings.filterwarnings(
        'ignore', message='Connection pool is full, discarding connection: 127.0.0.1'
    )
    requests.packages.urllib3.disable_warnings()
    driver.implicitly_wait(10)
    driver.set_page_load_timeout(20)
    driver.delete_all_cookies()

    return driver


# %%
# Function to current check window
def check_window(driver, main_window, window_before):

    new_window = False

    # If results open in new window, make sure everything loads
    window_num = len(driver.window_handles)

    if window_num > 1:
        new_window = True
        print(
            f'There are {window_num} windows open.\nLoading job details in new window.'
        )
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'JobView'))
            )
        except TimeoutException:
            print(f'Unexpected error loading new windows: {sys.exc_info()[0]}')
        try:
            window_after = [
                window for window in driver.window_handles if window != window_before
            ][0]
        except (TimeoutException, NoSuchWindowException, IndexError):
            print(
                f'Unexpected error opening result in new window: {sys.exc_info()[0]}.'
            )
        print(
            f'New window opened. Opening results in new window: {window_after}. {len(driver.window_handles)} windows opened so far.'
        )
        driver.switch_to.window(window_after)
    elif main_window == window_before:
        new_window = False
        print(f'No new window opened. Remaining on main window: {main_window}')

    return new_window


# %%
# Function to check which window to go go
def check_window_back(driver, main_window, window_before, new_window):

    # Go to main results page
    if new_window is True:
        # driver.current_window_handle =! main_window:
        print('Current window is not main window.')
        try:
            window_before = driver.window_handles[0]
            driver.switch_to.window(window_before)
            # Get html of page with beautifulsoup4
            html = driver.page_source
            print('Feeding html driver to BeautifulSoup.')
            soup = BeautifulSoup(html, 'lxml')
        except (NoSuchWindowException, NoSuchElementException):
            print(f'Could not go back to first Window: {window_before}')
        else:
            print(f'Going back to first window: {window_before}')

    elif new_window is False:
        pass


# %% [markdown]
# keywords_main_info

# %%
def remove_code(keywords_lst: list, keyword_clean_lst=None) -> list:

    if keyword_clean_lst is None:
        keyword_clean_lst = []

    for s in keywords_lst:
        lst = s.split()
        for i in lst:
            if len(i) <= 2:
                lst.remove(i)
            keyword_clean_lst.append(' '.join(lst))

    return [x for x in keyword_clean_lst if x != '']


# %%
# Function to clean keyword list
def clean_and_translate_keyword_list(
    keywords_lst: list,
    translate_enabled: bool = False,
    translator=Translator(),
) -> list:

    assert all(isinstance(i, str)
               for i in keywords_lst), 'Keywords must be strings.'

    # Collect all and and comma containing keywords
    and_comma = [i for i in keywords_lst if (',' in i) or ('and' in i)]

    # Remove ands and commas and append to keywords
    if len(and_comma) > 0:
        for i in and_comma:
            for x in re.split('and|,', i.strip().lower()):
                keywords_lst.append(x.strip().lower())

        # Remove duplicates
        keywords_lst = list(set(keywords_lst) ^ set(and_comma))

    else:
        keywords_lst = list(set(keywords_lst))

    # # Remove codes
    keywords_lst = remove_code(keywords_lst)

    # Singularize and remove duplicates
    keywords_list = list(
        set(
            list(
                map(
                    lambda line: (Word(line.lower()).singularize()).lower(),
                    keywords_lst,
                )
            )
        )
    )

    # Remove all non-specific keywords
    for i in keywords_list:
        if 'other ' in i.lower() and i.lower() not in ['other business support', 'other service activities']:
            keywords_list.append(i.lower().split('other')[1])
            keywords_list.remove(i)
        if ' (excl.' in i.lower():
            keywords_list.append(i.lower().split(' (excl.')[0].lower())
            keywords_list.remove(i)
        if '_(excl' in i.lower():
            keywords_list.append(i.lower().split('_(excl')[0].lower())
            keywords_list.remove(i)
    for i in keywords_list:
        if ' (' in i.lower():
            keywords_list.append(i.lower().split(' (')[0].lower())
            keywords_list.remove(i)
        if "-Noon's" in i.lower():
            keywords_list.append(i.lower().split('-Noon')[0].lower())
            keywords_list.remove(i)
        if len(i) <= 2:
            keywords_list.remove(i)
    for i in keywords_list:
        for w_keyword, r_keyword in keyword_trans_dict.items():
            if str(i.lower()) == w_keyword.lower():
                keywords_list.remove(i)
                keywords_list.append(r_keyword)

    # Remove duplicates
    keywords_list = list(filter(None, list(set(keywords_list))))

    # Translate to Dutch
    if translate_enabled is True:
        for english_keyword in keywords_list:
            while True:
                try:
                    dutch_keyword = translator.translate(english_keyword).text
                except Exception as e:
                    time.sleep(0.3)
                    continue
                break
            keywords_list.append(dutch_keyword.lower())

        # Remove duplicates
        keywords_list = list(filter(None, list(set(keywords_list))))

    return list(
        filter(None, list(set([i.lower().strip()
               for i in keywords_list if i])))
    )


# %%
def save_trans_keyword_list(trans_keyword_list, parent_dir=validate_path(f'{code_dir}/1. Scraping/CBS/Data/')):

    for keyword in trans_keyword_list:
        for w_keyword, r_keyword in keyword_trans_dict.items():
            if keyword.strip().lower() == w_keyword.strip().lower():
                trans_keyword_list.remove(keyword)
                trans_keyword_list.append(r_keyword.strip().lower())

    trans_keyword_list = clean_and_translate_keyword_list(list(
        set(
            list(
                map(
                    lambda keyword: (
                        Word(keyword.lower().strip()).singularize()).lower(),
                    trans_keyword_list,
                )
            )
        )
    ))

    with open(f'{parent_dir}trans_keyword_list.txt', 'w') as f:
        for i in set(trans_keyword_list):
            f.write(f'{i.lower()}\n')

    return trans_keyword_list


# %% [markdown]
# cbs_scraping
# %%
# Function to get translated and cleaned keyword list
def get_trans_keyword_list(parent_dir=validate_path(f'{code_dir}/1. Scraping/CBS/Data/')):

    with open(f'{parent_dir}trans_keyword_list.txt', 'r') as f:
        trans_keyword_list = [line.rstrip(' \n') for line in f]

    trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

    return trans_keyword_list


# %%
# # Function to read and save keyword lists
# def read_and_save_keyword_list(
#     print_enabled: bool = False,
#     save_enabled: bool = True,
#     translate_enabled: bool = False,
#     sectors_file_path: str = validate_path(
#         f'{code_dir}/1. Scraping/CBS/Found Data/'),
#     use_top10_data: bool = False,
#     age_limit: int = 45,
#     age_ratio: int = 10,
#     gender_ratio: int = 20,
# ):

#     if print_enabled is True:
#         print(
#             f'NOTE: The function "read_and_save_keyword_list" contains the following optional (default) arguments:\n{get_default_args(read_and_save_keyword_list)}'
#         )
#     # Augment Keywords List
#     # Gender
#     if use_top10_data is True:
#         # Highest % of women per occupation
#         keyword_file_path_womenvocc = validate_path(
#             f'{sectors_file_path}Top 10 highest % of women in occupations (2018).csv'
#         )
#         df_womenvocc = pd.read_csv(keyword_file_path_womenvocc)
#         #
#         _keywords_womenvocc = df_womenvocc['Beroep'].loc[1:].to_list()

#         # Highest % of men per occupation
#         keyword_file_path_menvocc = validate_path(
#             f'{sectors_file_path}Top 10 highest % of men in occupations (2018).csv'
#         )
#         df_menvocc = pd.read_csv(keyword_file_path_menvocc)
#         keywords_menvocc = df_menvocc['Beroep'].loc[1:].to_list()
#     elif use_top10_data is False:
#         keywords_womenvocc = []
#         keywords_menvocc = []

#     # Read into df
#     df_sectors = get_sector_df_from_cbs()
#     df_sectors.set_index(
#         ('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name'), inplace=True)

#     # Gender Sectors DFs
#     df_sector_gen_mixed = df_sectors.loc[df_sectors[(
#         'Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Mixed Gender']
#     df_sector_women = df_sectors.loc[df_sectors[(
#         'Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Female']
#     df_sector_men = df_sectors.loc[df_sectors[(
#         'Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Male']

#     # Make Mixed Gender keywords list
#     keywords_genvsect = df_sector_gen_mixed.index.to_list()
#     keywords_genvsect = clean_and_translate_keyword_list(
#         keywords_genvsect, translate_enabled)

#     # Add female and male sectors to lists
#     # Female Sectors + DF women v occ
#     for keywords_list in df_sector_women[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].to_list():
#         keywords_womenvocc.extend(keywords_list)
#         keywords_womenvocc.extend(df_sector_men.index.to_list())
#     keywords_womenvocc = clean_and_translate_keyword_list(
#         keywords_womenvocc, translate_enabled)

#     # Male Sectors + DF men v occ
#     for keywords_list in df_sector_men[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].to_list():
#         keywords_menvocc.extend(keywords_list)
#         keywords_menvocc.extend(df_sector_men.index.to_list())
#     keywords_menvocc = clean_and_translate_keyword_list(
#         keywords_menvocc, translate_enabled)

#     ################################################### AGE ###################################################
#     # Age Sectors DFs
#     df_sector_age_mixed = df_sectors.loc[df_sectors[(
#         'Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Mixed Age']
#     df_sector_old = df_sectors.loc[df_sectors[(
#         'Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Older']
#     df_sector_young = df_sectors.loc[df_sectors[(
#         'Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Younger']

#     # Make Mixed Age keywords list
#     keywords_agevsect = df_sector_age_mixed.index.to_list()
#     keywords_agevsect = clean_and_translate_keyword_list(
#         keywords_agevsect, translate_enabled)

#     # Add older and younger sectors to lists
#     # Older Sectors
#     keywords_oldvocc = df_sector_old.index.to_list()
#     keywords_oldvocc = clean_and_translate_keyword_list(
#         keywords_oldvocc, translate_enabled)

#     # Younger Sectors
#     keywords_youngvocc = df_sector_young.index.to_list()
#     keywords_youngvocc = clean_and_translate_keyword_list(
#         keywords_youngvocc, translate_enabled)

#     ################################################### SAVE ###################################################

#     # Print and save lists
#     if print_enabled is True:
#         print(
#             f'Female keywords total {len(keywords_womenvocc)}:\n{keywords_womenvocc}\n')
#         print(
#             f'Male keywords total {len(keywords_menvocc)}:\n{keywords_menvocc}\n')
#         print(
#             f'Mixed gender keywords total {len(keywords_genvsect)}:\n{keywords_genvsect}\n'
#         )
#         print(
#             f'Older worker keywords total {len(keywords_oldvocc)}:\n{keywords_oldvocc}\n')
#         print(
#             f'Younger keywords total {len(keywords_youngvocc)}:\n{keywords_youngvocc}\n'
#         )
#         print(
#             f'Mixed age keywords total {len(keywords_agevsect)}:\n{keywords_agevsect}\n'
#         )

#     keywords_dict_per_cat = {
#         'keywords_womenvocc': keywords_womenvocc,
#         'keywords_menvocc': keywords_menvocc,
#         'keywords_genvsect': keywords_genvsect,
#         'keywords_oldvocc': keywords_oldvocc,
#         'keywords_youngvocc': keywords_youngvocc,
#         'keywords_agevsect': keywords_agevsect,
#     }
#     if save_enabled is True:
#         with open(
#             f'{code_dir}/1. Scraping/CBS/Data/keywords_dict_per_cat.json', 'w', encoding='utf8'
#         ) as f:
#             json.dump(keywords_dict_per_cat, f)

#         for key, value in keywords_dict_per_cat.items():
#             if translate_enabled is False:
#                 save_path_file_name = f'Sectors List/{str(key)}.txt'
#             elif translate_enabled is True:
#                 save_path_file_name = (
#                     f'Sectors List/{str(key)}_with_nl.txt'
#                 )

#             if print_enabled is True:
#                 print(
#                     f'Saving {key} of length: {len(value)} to file location {sectors_file_path}.'
#                 )
#             with open(sectors_file_path + save_path_file_name, 'w') as f:
#                 for i in value:
#                     f.write(f'{i.lower()}\n')

#     elif save_enabled is False:
#         print('No keyword list save enabled.')

#     return (
#         keywords_dict_per_cat,
#         keywords_womenvocc,
#         keywords_menvocc,
#         keywords_genvsect,
#         keywords_oldvocc,
#         keywords_youngvocc,
#         keywords_agevsect,
#         df_sector_women,
#         df_sector_men,
#         df_sector_old,
#         df_sector_young,
#     )


# # %%
# def get_keywords_from_cbs(
#     save_enabled: bool = True,
#     keywords_file_path: str = f'{code_dir}/1. Scraping/CBS/Found Data/Sectors List/',
#     cols=['Industry class / branch (SIC2008)', 'Sex of employee',
#           'Other characteristics employee', 'Employment/Jobs (x 1 000)'],
#     age_limit: int = 45,
#     age_ratio: int = 10,
#     gender_ratio: int = 20,
# ):

#     keywords_dict_per_cat, keywords_womenvocc, keywords_menvocc, keywords_genvsect, keywords_oldvocc, keywords_youngvocc, keywords_agevsect, df_sector_women, df_sector_men, df_sector_old, df_sector_young = read_and_save_keyword_list()

#     df_sectors = get_sector_df_from_cbs()
#     df_sectors.set_index(
#         ('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name'), inplace=True)

#     # Make dfs, lists and dicts for each group
#     sectors_list = clean_and_translate_keyword_list(df_sectors.loc['Agriculture and industry': 'Other service activities', (
#         'SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].index.to_list())

#     female_sectors = df_sectors.loc[df_sectors[(
#         'Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Female']

#     female_list = clean_and_translate_keyword_list(
#         female_sectors.index.to_list())
#     female_dict = female_sectors.to_dict('index')

#     male_sectors = df_sectors.loc[df_sectors[(
#         'Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Male']
#     male_list = clean_and_translate_keyword_list(male_sectors.index.to_list())
#     male_dict = male_sectors.to_dict()

#     all_gender_sectors = pd.concat([female_sectors, male_sectors])
#     all_gender_list = clean_and_translate_keyword_list(
#         all_gender_sectors.index.to_list())
#     all_gender_dict = all_gender_sectors.to_dict()

#     old_sectors = df_sectors.loc[df_sectors[(
#         'Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Older']
#     old_list = clean_and_translate_keyword_list(old_sectors.index.to_list())
#     old_dict = old_sectors.to_dict()

#     young_sectors = df_sectors.loc[df_sectors[(
#         'Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Younger']
#     young_list = clean_and_translate_keyword_list(
#         young_sectors.index.to_list())
#     young_dict = young_sectors.to_dict()

#     all_age_sectors = pd.concat([old_sectors, young_sectors])
#     all_age_list = clean_and_translate_keyword_list(
#         all_age_sectors.index.to_list())
#     all_age_dict = all_age_sectors.to_dict()

#     # Save lists
#     if save_enabled is True:
#         with open(f'{keywords_file_path}keywords_sectors_FROM_SECTOR.txt', 'w') as f:
#             for i in sectors_list:
#                 f.write(f'{i.lower()}\n')

#         with open(
#             f'{keywords_file_path}keywords_womenvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
#             'w',
#         ) as f:
#             for i in female_list:
#                 f.write(f'{i.lower()}\n')

#         with open(
#             f'{keywords_file_path}keywords_menvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
#             'w',
#         ) as f:
#             for i in male_list:
#                 f.write(f'{i.lower()}\n')

#         with open(
#             f'{keywords_file_path}keywords_genvsect_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
#             'w',
#         ) as f:
#             for i in all_gender_list:
#                 f.write(f'{i.lower()}\n')

#         with open(
#             f'{keywords_file_path}keywords_oldvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
#             'w',
#         ) as f:
#             for i in old_list:
#                 f.write(f'{i.lower()}\n')

#         with open(
#             f'{keywords_file_path}keywords_youngvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
#             'w',
#         ) as f:
#             for i in young_list:
#                 f.write(f'{i.lower()}\n')

#         with open(
#             f'{keywords_file_path}keywords_agevsect_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
#             'w',
#         ) as f:
#             for i in all_age_list:
#                 f.write(f'{i.lower()}\n')

#     return (
#         df_sectors,
#         female_sectors,
#         male_sectors,
#         all_gender_sectors,
#         old_sectors,
#         young_sectors,
#         all_age_sectors,
#     )


# %%
# # Find file location
# def get_keyword_list(
#     print_enabled: bool = False,
#     get_from_cbs: bool = True,
#     age_limit=45,
#     age_ratio=10,
#     gender_ratio=20,
# ):

#     keywords_file_path = validate_path(
#         f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Analysis and Dataset Used/'
#     )

#     if get_from_cbs is True:
#         (
#             df_sectors,
#             female_sectors,
#             male_sectors,
#             all_gender_sectors,
#             old_sectors,
#             young_sectors,
#             all_age_sectors,
#         ) = get_keywords_from_cbs(
#             save_enabled=True,
#             age_limit=age_limit,
#             age_ratio=age_ratio,
#             gender_ratio=gender_ratio,
#         )

#     # Women Sector
#     keywords_dict_per_cat, keywords_womenvocc, keywords_menvocc, keywords_genvsect, keywords_oldvocc, keywords_youngvocc, keywords_agevsect, df_sector_women, df_sector_men, df_sector_old, df_sector_young = read_and_save_keyword_list()

#     with open(keywords_file_path + 'keywords_womenvocc.txt', 'r') as f:
#         keywords_womenvocc = f.read().splitlines()
#     with open(
#         keywords_file_path
#         + f'keywords_womenvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
#         'r',
#     ) as f:
#         keywords_womenvocc_sectors = f.read().splitlines()
#         keywords_womenvocc.extend(keywords_womenvocc_sectors)
#     if 'busines' in keywords_womenvocc:
#         keywords_womenvocc.remove('busines')
#     if 'busine' in keywords_womenvocc:
#         keywords_womenvocc.remove('busine')
#     keywords_womenvocc = list(filter(None, list(set(keywords_womenvocc))))
#     keywords_womenvocc = clean_and_translate_keyword_list(keywords_womenvocc)

#     # Men Sector
#     with open(keywords_file_path + 'keywords_menvocc.txt', 'r') as f:
#         keywords_menvocc = f.read().splitlines()
#     with open(
#         keywords_file_path
#         + f'keywords_menvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
#         'r',
#     ) as f:
#         keywords_menvocc_sectors = f.read().splitlines()
#         keywords_menvocc.extend(keywords_menvocc_sectors)
#     if 'busines' in keywords_menvocc:
#         keywords_menvocc.remove('busines')
#     if 'busine' in keywords_menvocc:
#         keywords_menvocc.remove('busine')
#     keywords_menvocc = list(filter(None, list(set(keywords_menvocc))))
#     keywords_menvocc = clean_and_translate_keyword_list(keywords_menvocc)

#     # Gender Segregated Sector
#     keywords_genvsect = keywords_womenvocc + keywords_menvocc
#     if 'busines' in keywords_genvsect:
#         keywords_genvsect.remove('busines')
#     if 'busine' in keywords_genvsect:
#         keywords_genvsect.remove('busine')
#     keywords_genvsect = list(filter(None, list(set(keywords_genvsect))))
#     keywords_genvsect = clean_and_translate_keyword_list(keywords_genvsect)

#     # Old worker Sector
#     with open(keywords_file_path + 'keywords_oldvocc.txt', 'r') as f:
#         keywords_oldvocc = f.read().splitlines()
#     with open(
#         keywords_file_path
#         + f'keywords_oldvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
#         'r',
#     ) as f:
#         keywords_oldvocc_sectors = f.read().splitlines()
#         keywords_oldvocc.extend(keywords_oldvocc_sectors)
#     if 'busines' in keywords_oldvocc:
#         keywords_oldvocc.remove('busines')
#     if 'busine' in keywords_oldvocc:
#         keywords_oldvocc.remove('busine')
#     keywords_oldvocc = list(filter(None, list(set(keywords_oldvocc))))
#     keywords_oldvocc = clean_and_translate_keyword_list(keywords_oldvocc)

#     # Young worker Sector
#     with open(keywords_file_path + 'keywords_youngvocc.txt', 'r') as f:
#         keywords_youngvocc = f.read().splitlines()
#     with open(
#         keywords_file_path
#         + f'keywords_youngvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
#         'r',
#     ) as f:
#         keywords_youngvocc_sectors = f.read().splitlines()
#         keywords_youngvocc.extend(keywords_youngvocc_sectors)
#     if 'busines' in keywords_youngvocc:
#         keywords_youngvocc.remove('busines')
#     if 'busine' in keywords_youngvocc:
#         keywords_youngvocc.remove('busine')
#     keywords_youngvocc = list(filter(None, list(set(keywords_youngvocc))))
#     keywords_youngvocc = clean_and_translate_keyword_list(keywords_youngvocc)

#     # Age Segregated Sector
#     keywords_agevsect = keywords_oldvocc + keywords_youngvocc
#     if 'busines' in keywords_agevsect:
#         keywords_agevsect.remove('busines')
#     if 'busine' in keywords_agevsect:
#         keywords_agevsect.remove('busine')
#     keywords_agevsect = list(filter(None, list(set(keywords_agevsect))))
#     keywords_agevsect = clean_and_translate_keyword_list(keywords_agevsect)

#     # All Sector
#     sbi_english_keyword_list, sbi_english_keyword_dict, sbi_sectors_dict, sbi_sectors_dict_full, sbi_sectors_dom_gen, sbi_sectors_dom_age, sbi_sectors_keywords_gen_dom, sbi_sectors_keywords_age_dom, sbi_sectors_keywords_full_dom, trans_keyword_list = get_sbi_sectors_list()
#     # keywords_sector = list(set([y for x in df_sectors.loc['Agriculture and industry': 'Other service activities', ('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].values.tolist() if isinstance(x, list) for y in x]))
#     keywords_sector = trans_keyword_list
#     with open(keywords_file_path + 'keywords_sectors_FROM_SECTOR.txt', 'r') as f:
#         keywords_sector_sectors = f.read().splitlines()
#     keywords_sector.extend(keywords_sector_sectors)

#     if 'busines' in keywords_sector:
#         keywords_sector.remove('busines')
#     if 'busine' in keywords_sector:
#         keywords_sector.remove('busine')
#     keywords_sector = list(filter(None, list(set(keywords_sector))))
#     keywords_sector = clean_and_translate_keyword_list(keywords_sector)

#     with open(keywords_file_path + 'keywords_sector.txt', 'w') as f:
#         for word in keywords_sector:
#             f.write(word + '\n')

#     keywords_list = (
#         keywords_sector
#         + keywords_womenvocc
#         + keywords_menvocc
#         + keywords_oldvocc
#         + keywords_youngvocc
#     )
#     if 'busines' in keywords_list:
#         keywords_list.remove('busines')
#     if 'busine' in keywords_list:
#         keywords_list.remove('busine')
#     # Remove duplicates
#     keywords_list = list(filter(None, list(set(keywords_list))))
#     keywords_list = clean_and_translate_keyword_list(keywords_list)

#     # Add mixed gender
#     # mixed_gender = [x for x in keywords_list if not x in keywords_womenvocc]
#     # mixed_gender = [x for x in mixed_gender if not x in keywords_menvocc]
#     # keywords_genvsect.extend(mixed_gender)
#     mixed_gender = [
#         k
#         for k in keywords_list
#         if (k not in keywords_womenvocc) and (k not in keywords_menvocc)
#     ]
#     if 'busines' in mixed_gender:
#         mixed_gender.remove('busines')
#     if 'busine' in mixed_gender:
#         mixed_gender.remove('busine')
#     mixed_gender = list(filter(None, list(set(mixed_gender))))
#     mixed_gender = clean_and_translate_keyword_list(mixed_gender)
#     mixed_age = [
#         k
#         for k in keywords_list
#         if (k not in keywords_oldvocc) and (k not in keywords_youngvocc)
#     ]
#     if 'busines' in mixed_age:
#         mixed_age.remove('busines')
#     if 'busine' in mixed_age:
#         mixed_age.remove('busine')
#     mixed_age = list(filter(None, list(set(mixed_age))))
#     mixed_age = clean_and_translate_keyword_list(mixed_age)

#     if print_enabled is True:
#         # Print and save lists
#         print(f'All sector total {len(keywords_sector)}:\n{keywords_sector}\n')
#         print(
#             f'Female keywords total {len(keywords_womenvocc)}:\n{keywords_womenvocc}\n'
#         )
#         print(
#             f'Male keywords total {len(keywords_menvocc)}:\n{keywords_menvocc}\n')
#         print(
#             f'Gender Segregated total {len(keywords_genvsect)}:\n{keywords_genvsect}\n'
#         )
#         print(f'Mixed Gender total {len(mixed_gender)}:\n{mixed_gender}\n')
#         print(
#             f'Older worker keywords total {len(keywords_oldvocc)}:\n{keywords_oldvocc}\n'
#         )
#         print(
#             f'Younger keywords total {len(keywords_youngvocc)}:\n{keywords_youngvocc}\n'
#         )
#         print(
#             f'Age Segregated total {len(keywords_agevsect)}:\n{keywords_agevsect}\n')
#         print(f'Mixed Age total {len(mixed_age)}:\n{mixed_age}\n')
#         print(f'All Keywords total {len(keywords_list)}:\n{keywords_list}')

#     return (
#         keywords_list,
#         keywords_sector,
#         keywords_womenvocc,
#         keywords_menvocc,
#         keywords_genvsect,
#         keywords_oldvocc,
#         keywords_youngvocc,
#         keywords_agevsect,
#     )

# %%
# Function to access args
def get_args(
    language='en',
    save_enabled=True,
    print_enabled=False,
    plots_enabled=False,
    excel_save=True,
    txt_save=True,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    columns_list=[
        'Job ID',
        'Sentence',
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    columns_fill_list=['Job ID', 'Sentence'],
    columns_drop_list=[
        'Search Keyword',
        'Gender',
        'Age',
        'Platform',
        'Job Title',
        'Company Name',
        'Location',
        'Rating',
        'Industry',
        'Sector',
        'Type of ownership',
        'Employment Type',
        'Seniority Level',
        'Company URL',
        'Job URL',
        'Job Age',
        'Job Age Number',
        'Job Date',
        'Collection Date',
    ],
    gender_sectors=['Female', 'Male', 'Mixed Gender'],
    age_sectors=['Older Worker', 'Younger', 'Mixed Age'],
    format_props={
        'border': 0,
        'font_name': 'Times New Roman',
        'font_size': 12,
        'font_color': 'black',
        'bold': True,
        'align': 'left',
        'text_wrap': True,
    },
    validation_props={'validate': 'list', 'source': [0, 1]},
    data_save_path=validate_path(f'{code_dir}/data/'),
    age_limit=45,
    age_ratio=10,
    gender_ratio=20,
    file_save_format='pkl',
    file_save_format_backup='csv',
    image_save_format='png',
):
    parent_dir = validate_path(
        f'{data_save_path}content analysis + ids + sectors/')
    content_analysis_dir = validate_path(f'{code_dir}2. Cleaning and Preprocessing/Coding Material/')
    df_dir = validate_path(f'{data_save_path}final dfs/')
    models_save_path = validate_path(f'{data_save_path}classification models/')
    table_save_path = validate_path(f'{data_save_path}output tables/')
    plot_save_path = validate_path(f'{data_save_path}plots/')
    embeddings_save_path = validate_path(f'{data_save_path}embeddings models/')

    # (
    #     keywords_list,
    #     keywords_sector,
    #     keywords_womenvocc,
    #     keywords_menvocc,
    #     keywords_genvsect,
    #     keywords_oldvocc,
    #     keywords_youngvocc,
    #     keywords_agevsect,
    # ) = get_keyword_list()

    # (
    #     sbi_english_keyword_list,
    #     sbi_english_keyword_dict,
    #     sbi_sectors_dict,
    #     sbi_sectors_dict_full,
    #     sbi_sectors_dom_gen,
    #     sbi_sectors_dom_age,
    #     sbi_sectors_keywords_gen_dom,
    #     sbi_sectors_keywords_age_dom,
    #     sbi_sectors_keywords_full_dom,
    #     trans_keyword_list
    # ) = get_sbi_sectors_list()

    return {
        'language': language,
        'save_enabled': save_enabled,
        'print_enabled': print_enabled,
        'plots_enabled': plots_enabled,
        'excel_save': excel_save,
        'txt_save': txt_save,
        'site_list': site_list,
        'columns_list': columns_list,
        'columns_fill_list': columns_fill_list,
        'columns_drop_list': columns_drop_list,
        'format_props': format_props,
        'validation_props': validation_props,
        'data_save_path': data_save_path,
        'parent_dir': parent_dir,
        'content_analysis_dir': content_analysis_dir,
        'df_dir': df_dir,
        'models_save_path': models_save_path,
        'table_save_path': table_save_path,
        'plot_save_path': plot_save_path,
        'embeddings_save_path': embeddings_save_path,
        'age_limit': age_limit,
        'age_ratio': age_ratio,
        'gender_ratio': gender_ratio,
        'file_save_format': file_save_format,
        'file_save_format_backup': file_save_format_backup,
        'image_save_format': image_save_format,
        # 'keywords_list': keywords_list,
        # 'keywords_sector': keywords_sector,
        # 'keywords_womenvocc': keywords_womenvocc,
        # 'keywords_menvocc': keywords_menvocc,
        # 'keywords_genvsect': keywords_genvsect,
        # 'keywords_oldvocc': keywords_oldvocc,
        # 'keywords_youngvocc': keywords_youngvocc,
        # 'keywords_agevsect': keywords_agevsect,
        # 'sbi_english_keyword_list': sbi_english_keyword_list,
        # 'sbi_english_keyword_dict': sbi_english_keyword_dict,
        # 'sbi_sectors_dict': sbi_sectors_dict,
        # 'sbi_sectors_dict_full': sbi_sectors_dict_full,
        # 'sbi_sectors_dom_gen': sbi_sectors_dom_gen,
        # 'sbi_sectors_dom_age': sbi_sectors_dom_age,
        # 'trans_keyword_list': trans_keyword_list,
    }


# %%
# Main Data
def main_info(keyword: str, site: str, save_path: str = validate_path(f'{main_dir}'), args=get_args()):

    save_path = validate_path(f'{scraped_data}/{site}/Data/')
    if site not in str(save_path):
        save_path = validate_path(f'{scraped_data}/{site}/Data/')

    if site.lower().strip() == 'indeed':
        keyword_url = '+'.join(keyword.lower().split(' '))
    elif site.lower().strip() == 'glassdoor':
        keyword_url = '-'.join(keyword.lower().split(' '))
    elif site.lower().strip() == 'linkedin':
        keyword_url = '%20'.join(keyword.lower().split(' '))
    elif not site:
        keyword_url = ''

    keyword_file = '_'.join(keyword.lower().split(' '))
    json_file_name = f'{site.lower()}_jobs_dict_{keyword_file.lower()}.json'.replace(
        "-Noon's MacBook Pro", '').replace('_(excl', '')
    df_file_name = f'{site.lower()}_jobs_df_{keyword_file.lower()}.{args["file_save_format_backup"]}'.replace(
        "-Noon's MacBook Pro", '').replace('_(excl', '')
    logs_file_name = f'{site.lower()}_jobs_logs_{keyword_file.lower()}.log'.replace(
        "-Noon's MacBook Pro", '').replace('_(excl', '')
    filemode = 'a+' if is_non_zero_file(save_path +
                                        logs_file_name.lower()) is True else 'w+'

    return (
        keyword_url,
        keyword_file,
        save_path,
        json_file_name,
        df_file_name,
        logs_file_name,
        filemode,
    )


# %% [markdown]
# set_threads

# %%
# Function to check for popups all the time
def popup_checker(driver, popup) -> None:
    while True:
        try:
            popup(driver)
        except AttributeError:
            time.sleep(10)
            print(f'Unexpected error with popup checker: {sys.exc_info()[0]}.')


# thread_popup_checker = multiprocessing.Process(target=popup_checker, args = (popup,))


# %%
# Function to check whether element is stale
def stale_element(driver) -> None:
    if pytest.raises(StaleElementReferenceException):
        driver.refresh()


# %%
def stale_element_checker(driver, stale_element) -> None:
    while True:
        try:
            stale_element(driver)
        except AttributeError:
            time.sleep(10)
            print(
                f'Unexpected error with stale element checker: {sys.exc_info()[0]}.')


# thread_stale_element_checker = multiprocessing.Process(target=stale_element_checker, args = (driver,stale_element,))


# %%
def act_cool(min_time: int, max_time: int, be_visibly_cool: bool = False) -> None:
    seconds = round(random.uniform(min_time, max_time), 2)
    if be_visibly_cool is True:
        logging.info(f'\tActing cool for {seconds} seconds'.format(**locals()))
    print(f'Acting cool for {seconds} seconds'.format(**locals()))
    time.sleep(seconds)


# %%
def act_cool_checker(act_cool) -> None:
    while True:
        try:
            act_cool(1, 10, be_visibly_cool=False)
        except AttributeError:
            time.sleep(10)
            print(
                f'Unexpected error with act cool checker: {sys.exc_info()[0]}.')


# thread_act_cool_checker = multiprocessing.Process(target = act_cool_checker, args = (act_cool,))

# %%
# Function to start popup threading
def popup_thread(driver, popup, popup_checker):
    thread_popup_checker = Thread(
        target=popup_checker,
        args=(
            driver,
            popup,
        ),
    )
    if not thread_popup_checker.is_alive():
        thread_popup_checker.start()


# %%
# Function to start popup threading
def stale_element_thread(driver, stale_element, stale_element_checker):
    thread_stale_element_checker = Thread(
        target=stale_element_checker,
        args=(
            driver,
            stale_element,
        ),
    )
    if not thread_stale_element_checker.is_alive():
        thread_stale_element_checker.start()


# %%
# Function to start popup threading
def act_cool_thread(driver, act_cool, act_cool_checker):
    thread_act_cool_checker = multiprocessing.Process(
        target=act_cool_checker,
        args=(
            driver,
            act_cool,
        ),
    )
    if not thread_act_cool_checker.is_alive():
        thread_act_cool_checker.start()


# %% [markdown]
# scrape_support

# %%
# Function to convert bs4 to xpath
def xpath_soup(element, components=None):

    components = []
    child = element if element.name else element.parent
    for parent in child.parents:

        previous = itertools.islice(
            parent.children, 0, parent.contents.index(child))
        xpath_tag = child.name
        xpath_index = sum(1 for i in previous if i.name == xpath_tag) + 1
        components.append(
            xpath_tag if xpath_index == 1 else f'{xpath_tag}[{xpath_index}]'
        )
        child = parent
    components.reverse()

    return f'/{"/".join(components)}'


# %%
# Check if already collected
def id_check(
    driver,
    main_window,
    window_before,
    keyword,
    site,
    df_old_jobs,
    jobs,
    job_soup,
    job_id,
    jobs_count,
    save_path,
    json_file_name,
    job_present=False,
):
    if not df_old_jobs.empty:
        df_old_jobs = clean_df(df_old_jobs)
        type_check = np.all([isinstance(val, str)
                            for val in df_old_jobs['Job ID']])
        typ, typ_str = (str, 'str') if type_check == True else (int, 'int')

        if any(typ(job_id) == df_old_jobs['Job ID']):
            common_id = (
                df_old_jobs['Job ID']
                .loc[df_old_jobs['Job ID'].astype(typ_str) == typ(job_id)]
                .values[0]
            )
            print(
                f'This job ad with id {job_id} (df id = {common_id}) has already been collected to df. Moving to next.'
            )
            job_present = True
        elif all(typ(job_id) != df_old_jobs['Job ID']):
            job_present = False

    elif df_old_jobs.empty:
        if len(jobs) != 0:
            # jobs = remove_dupe_dicts(jobs)
            type_check = np.all(
                [isinstance(jobs_dict['Job ID'], str) for jobs_dict in jobs]
            )
            typ, typ_str = (str, 'str') if type_check == True else (int, 'int')

            if any(typ(job_id) == typ(job_dicts['Job ID']) for job_dicts in jobs):
                common_id = [
                    job_dicts['Job ID']
                    for job_dicts in jobs
                    if job_dicts['Job ID'] == typ(job_id)
                ]
                print(
                    f'This job ad with id {job_id} (df id = {common_id}) has already been collected to json. Moving to next.'
                )
                job_present = True
            elif not any(typ(job_id) == typ(job_dicts['Job ID']) for job_dicts in jobs):
                job_present = False

        elif len(jobs) == 0:
            job_present = False

    return jobs, job_present

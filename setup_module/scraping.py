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
# import os
# import sys
# from pathlib import Path
# code_dir = None
# code_dir_name = 'Code'
# unwanted_subdir_name = 'Analysis'
# for _ in range(5):
#     parent_path = str(Path.cwd().parents[_]).split('/')[-1]
#     if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):
#         code_dir = str(Path.cwd().parents[_])
#         if code_dir is not None:
#             break
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
import os
import sys
from pathlib import Path

code_dir = None
code_dir_name = 'Code'
unwanted_subdir_name = 'Analysis'

for _ in range(5):

    parent_path = str(Path.cwd().parents[_]).split('/')[-1]

    if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):

        code_dir = str(Path.cwd().parents[_])

        if code_dir is not None:
            break

main_dir = str(Path(code_dir).parents[0])
scraped_data = f'{code_dir}/scraped_data'
sys.path.append(code_dir)

# %%
from setup_module.imports import *

warnings.filterwarnings('ignore', category=DeprecationWarning)

# %%
def get_main_path(
    code_dir = None,
    code_dir_name = 'Code',
    unwanted_subdir_name = 'Analysis',
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
    for _ in range(5):

        parent_path = str(Path.cwd().parents[_]).split('/')[-1]

        if (code_dir_name in parent_path) and (unwanted_subdir_name not in parent_path):

            code_dir = str(Path.cwd().parents[_])

            if code_dir is not None:
                break

    main_dir = str(Path(code_dir).parents[0])

    return code_dir, main_dir, errors


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
def validate_path(file: str, file_extensions=['.*', 'chromedriver']) -> str:

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
def remove_dupe_dicts(l, jobs = None, seen_ad = None, seen_id = None):
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
def select_driver(webdriver_path = f'{code_dir}/setup_module/WebDriver/') -> str:

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

    # Proxy
    if proxy_enabled is True:
        print('Proxy browsing enabled.')
        ua = UserAgent()
        userAgent = ua.random
        req_proxy = RequestProxy()
        proxies = req_proxy.get_proxy_list()
        PROXY = proxies[5].get_address()
        print(f'Proxy country: {proxies[5].country}')
        caps['proxy'] = {
            'httpProxy': PROXY,
            'ftpProxy': PROXY,
            'sslProxy': PROXY,
            'proxyType': 'MANUAL',
        }
        options.add_argument(f'--proxy-server= {PROXY}')
        options.add_argument(f'user-agent={userAgent}')
    elif proxy_enabled is False:
        print('No proxy browsing enabled.')
        pass

    try:
        driver = webdriver.Chrome(
            executable_path=select_driver(),
            options=options,
            desired_capabilities=caps,
            # service_args=[f'--verbose", "--log-path={MyWriter.LOGS_PATH}'],
        )
    except Exception as e:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
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

    return keyword_clean_lst


# %%
# Function to clean keyword list
def clean_and_translate_keyword_list(
    keywords_lst: list,
    translate_enabled: bool = False,
    translator = Translator(),
) -> list:

    assert all(isinstance(i, str) for i in keywords_lst), 'Keywords must be strings.'

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
        filter(None, list(set([i.lower().strip() for i in keywords_list if i])))
    )


# %%
def save_trans_keyword_list(trans_keyword_list, parent_dir=validate_path(f'{code_dir}/scraped_data/CBS/Data/')):

    for keyword in trans_keyword_list:
        for w_keyword, r_keyword in keyword_trans_dict.items():
            if keyword.strip().lower() == w_keyword.strip().lower():
                trans_keyword_list.remove(keyword)
                trans_keyword_list.append(r_keyword.strip().lower())

    trans_keyword_list = clean_and_translate_keyword_list(list(
        set(
            list(
                map(
                    lambda keyword: (Word(keyword.lower().strip()).singularize()).lower(),
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
def get_trans_keyword_list(parent_dir=validate_path(f'{code_dir}/scraped_data/CBS/Data/')):

    with open(f'{parent_dir}trans_keyword_list.txt', 'r') as f:
        trans_keyword_list = [line.rstrip(' \n') for line in f]

    trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

    return trans_keyword_list

# %% Function to get sbi_sectors_dict
def get_sbi_sectors_list(
    save_enabled=True,
    parent_dir=validate_path(f'{code_dir}/scraped_data/CBS/'),
    ):

    sib_5_loc = validate_path(f'{parent_dir}Found Data/SBI_ALL_NACE_REV2.csv')
    data_save_dir = validate_path(f'{parent_dir}Data/')
    trans_keyword_list = get_trans_keyword_list()

    df_sbi_sectors = pd.read_csv(sib_5_loc, delimiter=',')
    df_sbi_sectors.columns = df_sbi_sectors.columns.str.strip()
    df_sbi_sectors.rename(columns = {'Description': 'Old_Sector_Name'}, inplace=True)
    df_sbi_sectors = df_sbi_sectors.dropna(subset=['Old_Sector_Name', 'Code'])
    df_sbi_sectors['Old_Sector_Name'] = df_sbi_sectors['Old_Sector_Name'].apply(lambda x: x.lower().strip())
    df_sbi_sectors = df_sbi_sectors.loc[df_sbi_sectors['Level'] == 1]
    df_sbi_sectors.drop(columns=['Level', 'Parent', 'This item includes', 'This item also includes', 'Rulings', 'This item excludes', 'Reference to ISIC Rev. 4'], inplace=True)

    df_sectors_all = pd.read_pickle(f'{data_save_dir}Sectors Output from script.pkl')[[('SBI Sector Titles'), ('Gender'), ('Age')]].droplevel('Categories', axis='columns')[[('SBI Sector Titles', 'Code'), ('SBI Sector Titles', 'Sector Name'), ('SBI Sector Titles', 'Keywords'), ('Gender', 'Dominant Category'), ('Age', 'Dominant Category')]].droplevel('Variables', axis='columns')
    df_sectors_all.columns = ['Code', 'Sector Name', 'Keywords', 'Gender Dominant Category', 'Age Dominant Category']
    df_sbi_sectors = df_sbi_sectors.merge(df_sectors_all, how='inner', on='Code')
    df_sbi_sectors.rename(columns = {'Sector Name': 'Sector_Name', 'Keywords': 'Used_Sector_Keywords', 'Gender Dominant Category': 'Gender_Dominant_Category', 'Age Dominant Category': 'Age_Dominant_Category'}, inplace=True)
    df_sbi_sectors['Sector_Name'] = df_sbi_sectors['Sector_Name'].apply(lambda x: x.strip().lower() if isinstance(x, str) else np.nan)
    df_sbi_sectors['Used_Sector_Keywords'] = df_sbi_sectors['Used_Sector_Keywords'].apply(lambda x: clean_and_translate_keyword_list(x) if isinstance(x, list) else np.nan)
    df_sbi_sectors.set_index(df_sbi_sectors['Code'], inplace=True)

    df_sbi_sectors.to_csv(f'{data_save_dir}SBI-5_Sectors.csv', index=True)
    df_sbi_sectors.to_excel(f'{data_save_dir}SBI-5_Sectors.xlsx', index=True)
    df_sbi_sectors.to_pickle(f'{data_save_dir}SBI-5_Sectors.pkl')

    sbi_english_keyword_list = [i for index, row in df_sbi_sectors['Used_Sector_Keywords'].iteritems() if isinstance(row, list) for i in row]
    sbi_english_keyword_list = clean_and_translate_keyword_list(sbi_english_keyword_list)

    if len(list(set(trans_keyword_list) - set(sbi_english_keyword_list))) > 0:
        trans_keyword_list = clean_and_translate_keyword_list(trans_keyword_list)
        if len(list(set(trans_keyword_list) - set(sbi_english_keyword_list))) > 0:
            trans_keyword_list = save_trans_keyword_list(trans_keyword_list)
            print(f'Unknown keyword found. Check trans_keyword_list for {list(set(trans_keyword_list) - set(sbi_english_keyword_list))} len {len(list(set(trans_keyword_list) - set(sbi_english_keyword_list)))}.')

    sbi_english_keyword_dict = df_sbi_sectors['Used_Sector_Keywords'].to_dict()
    sbi_sectors_dict = df_sbi_sectors.to_dict('index')
    sbi_sectors_dict_full = {}
    sbi_sectors_dom_gen = {}
    sbi_sectors_dom_age = {}
    sbi_sectors_keywords_gen_dom = defaultdict(list)
    sbi_sectors_keywords_age_dom = defaultdict(list)
    sbi_sectors_keywords_full_dom = defaultdict(list)
    for index, row in df_sbi_sectors.iterrows():
        sbi_sectors_dict_full[row['Sector_Name']] = row['Used_Sector_Keywords']
        sbi_sectors_dom_gen[row['Sector_Name']] = row['Gender_Dominant_Category']
        sbi_sectors_dom_age[row['Sector_Name']] = row['Age_Dominant_Category']
    for cat_keywords in df_sbi_sectors[['Gender_Dominant_Category', 'Used_Sector_Keywords']].to_dict(orient='split')['data']:
        sbi_sectors_keywords_gen_dom[cat_keywords[0]].extend(cat_keywords[1])
    for cat_keywords in df_sbi_sectors[['Age_Dominant_Category', 'Used_Sector_Keywords']].to_dict(orient='split')['data']:
        sbi_sectors_keywords_age_dom[cat_keywords[0]].extend(cat_keywords[1])
    for d in (sbi_sectors_keywords_gen_dom, sbi_sectors_keywords_age_dom):
        sbi_sectors_keywords_full_dom.update(d)

    if save_enabled is True:
        with open(f'{data_save_dir}sbi_english_keyword_list.txt', 'w', encoding='utf8') as f:
            for i in sbi_english_keyword_list:
                f.write(f'{i.lower()}\n')
        with open(f'{data_save_dir}sbi_english_keyword_dict.json', 'w', encoding='utf8') as f:
            json.dump(sbi_english_keyword_dict, f)
        with open(f'{data_save_dir}sbi_sectors_dict.json', 'w', encoding='utf8') as f:
            json.dump(sbi_sectors_dict, f)

        with open(f'{data_save_dir}sbi_sectors_keywords_gen_dom.json', 'w', encoding='utf8') as f:
            json.dump(sbi_sectors_keywords_gen_dom, f)
        with open(f'{data_save_dir}sbi_sectors_keywords_age_dom.json', 'w', encoding='utf8') as f:
            json.dump(sbi_sectors_keywords_age_dom, f)
        with open(f'{data_save_dir}sbi_sectors_keywords_full_dom.json.json', 'w', encoding='utf8') as f:
            json.dump(sbi_sectors_keywords_full_dom, f)

    return sbi_english_keyword_list, sbi_english_keyword_dict, sbi_sectors_dict, sbi_sectors_dict_full, sbi_sectors_dom_gen, sbi_sectors_dom_age, sbi_sectors_keywords_gen_dom, sbi_sectors_keywords_age_dom, sbi_sectors_keywords_full_dom, trans_keyword_list


# %% CBS Data request
def get_cbs_odata(
    sectors_file_path: str = validate_path(f'{code_dir}/scraped_data/CBS/Found Data/'),
    table_url='https://opendata.cbs.nl/ODataAPI/OData/',
    table_id='81434ENG',
    addition_url='/UntypedDataSet',
    select=['SexOfEmployee', 'TypeOfEmploymentContract', 'OtherCharacteristicsEmployee', 'IndustryClassBranchSIC2008', 'Periods', 'Jobs_1'],
):
    # data: https://opendata.cbs.nl/#/CBS/en/dataset/81434ENG/table?ts=1663627369191
    # instruction: https://data.overheid.nl/dataset/410-bevolking-op-eerste-van-de-maand--geslacht--leeftijd--migratieachtergrond
    # github: https://github.com/statistiekcbs/CBS-Open-Data-v4

    tables = cbsodata.get_table_list()
    for table in tables:
        if table['Identifier'] == table_id:
            data_info = table
    info = cbsodata.get_info(table_id)
    diffs = list(set(info.keys()) - set(data_info.keys()))
    for i in diffs:
        data_info[i] = info[i]

    with open(f'{sectors_file_path}cbs_data_info.json', 'w', encoding='utf8') as f:
        json.dump(data_info, f)

    dimensions = defaultdict(dict)
    for sel in select:
        if sel != 'Jobs_1':
            meta_data = pd.DataFrame(cbsodata.get_meta('81434ENG', sel))
        if sel == 'TypeOfEmploymentContract':
            meta_data = meta_data.loc[~meta_data['Title'].str.contains('Type of employment contract:')]
        if sel == 'OtherCharacteristicsEmployee':
            meta_data = meta_data.loc[~meta_data['Key'].str.contains('NAT')]
        if sel == 'Periods':
            meta_data = meta_data.loc[meta_data['Title'].astype(str) == '2020']

        for title, key in zip(meta_data['Title'].tolist(), meta_data['Key'].tolist()):
            if sel != 'Jobs_1':
                dimensions[sel][title] = key
    with open(f'{sectors_file_path}cbs_data_dimensions.json', 'w', encoding='utf8') as f:
        json.dump(dimensions, f)

    while True:
        try:
            data = pd.DataFrame(cbsodata.get_data(table_id, select=select))
            break
        except ConnectionError:
            time.sleep(5)

    data = data.loc[~data['TypeOfEmploymentContract'].str.contains('Type of employment contract:') & ~data['OtherCharacteristicsEmployee'].str.contains('Nationality:') & data['Periods'].str.contains('2020')]

    data.to_csv(f'{sectors_file_path}Sectors Tables/FINAL/Gender x Age_CBS_DATA_from_code.csv')

    # target_url = table_url + table_id + addition_url

    # data = pd.DataFrame()
    # while target_url:
    #     r = requests.get(target_url).json()
    #     data = data.append(pd.DataFrame(r['value']))

    #     if '@odata.nextLink' in r:
    #         target_url = r['@odata.nextLink']
    #     else:
    #         target_url = None

    return data


# %%
def save_sector_excel(
    df_sectors_all,
    tables_file_path,
    sheet_name='All',
    excel_file_name = 'Sectors Output from script.xlsx',
    age_limit: int = 45,
    age_ratio: int = 10,
    gender_ratio: int = 20,
):
    writer = pd.ExcelWriter(f'{tables_file_path}{excel_file_name}', engine='xlsxwriter')
    df_sectors_all.to_excel(writer, sheet_name=sheet_name, merge_cells = True, startrow = 3)
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]
    worksheet.set_row(6, None, None, {'hidden': True})
    worksheet.set_column(0, 0, None, None, {'hidden': True})
    # Title
    worksheet.merge_range(0, 0, 0, df_sectors_all.shape[1], 'Table 10', workbook.add_format({'bold': True, 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'left'}))
    worksheet.merge_range(1, 0, 1, df_sectors_all.shape[1], 'Sectoral Gender and Age Composition and Segregation, Keywords, Counts, and Percentages', workbook.add_format({'italic': True, 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'left'}))
    worksheet.merge_range(2, 0, 2, df_sectors_all.shape[1], 'Jobs Count per Sector (x 1000)', workbook.add_format({'bold': False, 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'center', 'top': True, 'bottom': True}))
    # Format column headers
    for col_num, value in enumerate(df_sectors_all.columns.values):
        for i in range(3):
            worksheet.write(3 + i, col_num + 1, value[i], workbook.add_format({'bold': False, 'font_name': 'Times New Roman', 'font_size': 12, 'font_color': 'black', 'align': 'center', 'top': True, 'bottom': True}))
            if value[i] == 'n':
                worksheet.set_column(col_num + 1, col_num + 1, 5.5)
            elif value[i] == 'Code':
                worksheet.set_column(col_num + 1, col_num + 1, 4.5)
            elif value[i] == 'Sector Name':
                worksheet.set_column(col_num + 1, col_num + 1, 28.5)
            elif value[i] == 'Keywords':
                worksheet.set_column(col_num + 1, col_num + 1, 30)
            elif value[i] == 'Keywords Count':
                worksheet.set_column(col_num + 1, col_num + 1, 13.5)
            elif value[i] == '% per Sector':
                worksheet.set_column(col_num + 1, col_num + 1, 12)
            elif value[i] == '% per Social Category':
                worksheet.set_column(col_num + 1, col_num + 1, 19.5)
            elif value[i] == '% per Workforce':
                worksheet.set_column(col_num + 1, col_num + 1, 15.5)
            elif value[i] == 'Dominant Category':
                worksheet.set_column(col_num + 1, col_num + 1, 24.5)
            elif value[i] == '% Sector per Workforce':
                worksheet.set_column(col_num + 1, col_num + 1, 21.5)

    # Borders
    perc = [col_num for col_num, value in enumerate(df_sectors_all.columns.values) if '%' in value[-1]]
    num = [col_num for col_num, value in enumerate(df_sectors_all.columns.values) if 'n' in value[-1]]
    word = [col_num for col_num, value in enumerate(df_sectors_all.columns.values) if value[-1] in ['Code', 'Sector Name', 'Dominant Category']]
    keyword = [col_num for col_num, value in enumerate(df_sectors_all.columns.values) if 'Keywords' in value[-1]]

    row_idx, col_idx = df_sectors_all.shape
    for c in range(col_idx):
        for r in range(row_idx):
            if c in perc:
                formats = {'num_format': '0.00%', 'font_name': 'Times New Roman', 'font_size': 12}
                if r == row_idx-1:
                    formats.update({'top': True, 'bottom': True})
            elif c in num:
                formats = {'num_format': '0', 'font_name': 'Times New Roman', 'font_size': 12}
                if r == row_idx-1:
                    formats.update({'top': True, 'bottom': True})
            elif c in word:
                formats = {'font_name': 'Times New Roman', 'font_size': 12}
                if r == row_idx-1:
                    formats.update({'top': True, 'bottom': True})
            elif c in keyword:
                formats = {'font_name': 'Times New Roman', 'font_size': 12, 'align': 'left'}
                if r == row_idx-1:
                    formats.update({'top': True, 'bottom': True})
            try:
                worksheet.write(r + 7, c + 1, df_sectors_all.iloc[r, c], workbook.add_format(formats))
            except TypeError:
                if isinstance(df_sectors_all.iloc[r, c], list):
                    value = str(df_sectors_all.iloc[r, c])
                else:
                    value = ''
                worksheet.write(r + 7, c + 1, value, workbook.add_format(formats))

    worksheet.merge_range(len(df_sectors_all)+7, 0, len(df_sectors_all)+7, df_sectors_all.shape[1], 'Note.', workbook.add_format({'italic': True, 'font_name': 'Times New Roman', 'font_size': 10, 'font_color': 'black', 'align': 'left'}))
    worksheet.merge_range(len(df_sectors_all)+8, 0, len(df_sectors_all)+8, df_sectors_all.shape[1], f'Threshold for gender = {df_sectors_all.loc[df_sectors_all.index[-1], ("Gender", "Female", "% per Workforce")]:.2f}% ± 20%', workbook.add_format({'italic': True, 'font_name': 'Times New Roman', 'font_size': 10, 'font_color': 'black', 'align': 'left'}))
    worksheet.merge_range(len(df_sectors_all)+9, 0, len(df_sectors_all)+9, df_sectors_all.shape[1], f'Threshold for age = {df_sectors_all.loc[df_sectors_all.index[-1], ("Age", f"Older (>= {age_limit} years)", "% per Workforce")]:.2f}% ± 10%', workbook.add_format({'italic': True, 'font_name': 'Times New Roman', 'font_size': 10, 'font_color': 'black', 'align': 'left'}))
    worksheet.merge_range(len(df_sectors_all)+10, 0, len(df_sectors_all)+10, df_sectors_all.shape[1], 'Source: Centraal Bureau voor de Statistiek (CBS)', workbook.add_format({'italic': True, 'font_name': 'Times New Roman', 'font_size': 8, 'font_color': 'black', 'align': 'left'}))

    writer.close()

# %%
# Function to get sector df from cbs
def get_sector_df_from_cbs(
    save_enabled: bool = True,
    parent_dir=validate_path(f'{code_dir}/scraped_data/CBS/'),
    cols = ['Industry class / branch (SIC2008)', 'Sex of employee', 'Other characteristics employee', 'Employment/Jobs (x 1 000)'],
    get_cbs_odata_enabled=False,
    age_limit: int = 45,
    age_ratio: int = 10,
    gender_ratio: int = 20,
):

    sectors_file_path: str = validate_path(f'{parent_dir}Found Data/')
    data_save_dir: str = validate_path(f'{parent_dir}Data/')

    # with open(f'{code_dir}/data/content analysis + ids + sectors/sbi_sectors_dict.json', 'r', encoding='utf8') as f:
    #     sbi_sectors_dict = json.load(f)
    sbi_english_keyword_list, sbi_english_keyword_dict, sbi_sectors_dict, sbi_sectors_dict_full, sbi_sectors_dom_gen, sbi_sectors_dom_age, sbi_sectors_keywords_gen_dom, sbi_sectors_keywords_age_dom, sbi_sectors_keywords_full_dom, trans_keyword_list = get_sbi_sectors_list()

    if get_cbs_odata_enabled is True:
        select = ['SexOfEmployee', 'TypeOfEmploymentContract', 'OtherCharacteristicsEmployee', 'IndustryClassBranchSIC2008', 'Periods', 'Jobs_1']
        odata_colnames_normalized = {'IndustryClassBranchSIC2008': 'Industry class / branch (SIC2008)', 'SexOfEmployee': 'Sex of employee', 'OtherCharacteristicsEmployee': 'Other characteristics employee', 'Jobs_1': 'Employment/Jobs (x 1 000)'}
        df_sectors = get_cbs_odata()
        df_sectors.rename(columns=odata_colnames_normalized, inplace=True)
    elif get_cbs_odata_enabled is False:
        # print(f'Error getting data from CBS Statline OData. Using the following file:\n{sectors_file_path}Sectors Tables/FINAL/Gender x Age_CBS_DATA.csv')
        # Read, clean, create code variable
        try:
            df_sectors = pd.read_csv(f'{sectors_file_path}Sectors Tables/FINAL/Gender x Age_CBS_DATA.csv', delimiter=';')
        except Exception:
            df_sectors = pd.read_csv(f'{sectors_file_path}Sectors Tables/FINAL/Gender x Age_CBS_DATA_from_code.csv', delimiter=';')


    df_sectors = df_sectors[cols]
    df_sectors.rename({'Sex of employee': 'Gender', 'Other characteristics employee': 'Age Range (in years)', 'Industry class / branch (SIC2008)': 'Sector Name', 'Employment/Jobs (x 1 000)': 'n'}, inplace=True, axis = 1)
    df_sectors.insert(0, 'Code', df_sectors['Sector Name'].apply(lambda row: row[0]))
    df_sectors['Sector Name'] = df_sectors['Sector Name'].apply(lambda row: row[2:].strip() if '-' not in row else row[3:].strip())

    # Categorize by age label
    all_age = df_sectors['Age Range (in years)'].unique().tolist()[1:]
    for i, word in enumerate(all_age):
        if word.startswith(str(age_limit)):
            young = all_age[:i]
            old = all_age[i:]
    conditions = [
        (df_sectors['Age Range (in years)'].isin(old)),
        (df_sectors['Age Range (in years)'].isin(young))
    ]
    choices = [f'Older (>= {age_limit} years)', f'Younger (< {age_limit} years)']
    age_cat = np.select(conditions, choices, default='Total')
    df_sectors.insert(3, 'Age', age_cat)
    choices.append('Total')
    df_sectors['Age'].astype('category').cat.reorder_categories(choices, inplace=True)

    # Change gender label
    df_sectors['Gender'].replace({'Sex: Female': 'Female', 'Sex: Male': 'Male'}, inplace=True)
    df_sectors['Gender'].astype('category').cat.reorder_categories(['Female', 'Male', 'Total'], inplace=True)

    # Rearrgane columns
    # Gender
    df_gender_only = df_sectors.pivot_table(values='n', index=['Code', 'Sector Name', 'Age'], columns=['Gender'], aggfunc='sum')
    df_gender_only.reset_index(inplace=True)
    df_gender_only = df_gender_only.loc[df_gender_only['Age'] == 'Total']
    df_gender_only.drop(columns=['Age', 'Total'], inplace=True)
    df_gender_only.reset_index(drop=True, inplace=True)
    df_gender_only.name = 'Gender'
    # Age
    df_age_only = df_sectors.pivot_table(values='n', index=['Code', 'Sector Name', 'Gender'], columns=['Age'], aggfunc='sum')
    df_age_only.reset_index(inplace=True)
    df_age_only = df_age_only.loc[df_age_only['Gender'] == 'Total']
    df_age_only.drop(columns=['Gender', 'Total'], inplace=True)
    df_age_only.reset_index(drop=True, inplace=True)
    df_age_only.name = 'Age'

    # Total
    df_total_only = df_sectors.pivot_table(values='n', index=['Code', 'Sector Name', 'Gender', 'Age'], aggfunc='sum')
    df_total_only.reset_index(inplace=True)
    df_total_only = df_total_only.loc[(df_total_only['Gender'] == 'Total') & (df_total_only['Age'] == 'Total')]
    df_total_only.drop(columns=['Gender', 'Age'], inplace=True)
    df_total_only.reset_index(drop=True, inplace=True)
    df_total_only.rename(columns={'n': 'Total Workforce'}, inplace=True)
    df_total_only.name = 'Total'

    # Merge all
    df_sectors_all = pd.merge(pd.merge(df_gender_only, df_age_only, how='outer'), df_total_only, how='outer')
    df_sectors_all.reset_index(inplace=True, drop=True)

    # Take out "All economic activities" row
    au = df_sectors_all.loc[df_sectors_all['Sector Name'] == 'All economic activities']
    au.loc[au['Code'] != 'A-U', 'Code'] = 'A-U'
    df_sectors_all = df_sectors_all[df_sectors_all['Sector Name'] != 'All economic activities']
    df_sectors_all.reset_index(inplace=True, drop=True)
    df_sectors_all = df_sectors_all.groupby(['Code'], as_index=True).agg({'Sector Name': 'first', **dict.fromkeys(df_sectors_all.loc[:, ~df_sectors_all.columns.isin(['Code', 'Sector Name'])].columns.to_list(), 'sum')})
    df_sectors_all.reset_index(inplace=True)

    # Add keywords
    df_sectors_all.insert(2, 'Keywords', df_sectors_all['Code'].apply(lambda row: sbi_sectors_dict[row]['Used_Sector_Keywords'] if row in sbi_sectors_dict and isinstance(row, str) else np.nan))
    df_sectors_all['Keywords'] = df_sectors_all['Keywords'].apply(lambda row: clean_and_translate_keyword_list(row) if isinstance(row, list) else np.nan)
    df_sectors_all.insert(3, 'Keywords Count', df_sectors_all['Keywords'].apply(lambda row: int(len(row)) if isinstance(row, list) else np.nan))

    # Add totals in bottom row
    df_sectors_all.loc[df_sectors_all[df_sectors_all['Sector Name'] == 'Other service activities'].index.values.astype(int)[0]+1, 'Sector Name'] = 'Total (excluding A-U)'
    df_sectors_all.iloc[df_sectors_all[df_sectors_all['Sector Name'] == 'Total (excluding A-U)'].index.values.astype(int)[0], ~df_sectors_all.columns.isin(['Code', 'Sector Name', 'Keywords'])] = df_sectors_all.sum(numeric_only=True)
    df_sectors_all.columns = pd.MultiIndex.from_tuples([('Industry class / branch (SIC2008)', 'Code'), ('Industry class / branch (SIC2008)', 'Sector Name'), ('Industry class / branch (SIC2008)', 'Keywords'), ('Industry class / branch (SIC2008)', 'Keywords Count'), ('Female', 'n'), ('Male', 'n'), (f'Older (>= {age_limit} years)', 'n'), (f'Younger (< {age_limit} years)', 'n'), ('Total Workforce', 'n')], names = ['Social category', 'Counts'])

    # Make percentages
    for index, row in df_sectors_all.iteritems():
        if ('Total' not in index[0]) and ('%' not in index[1]) and ('n' in index[1]) and (not isinstance(row[0], str)) and (not math.isnan(row[0])):
            df_sectors_all[(index[0], '% per Sector')] = row/df_sectors_all[('Total Workforce', 'n')]#*100
            df_sectors_all[(index[0], '% per Social Category')] = row/df_sectors_all.loc[df_sectors_all[df_sectors_all[('Industry class / branch (SIC2008)', 'Sector Name')] == 'Total (excluding A-U)'].index.values.astype(int)[0], index]#*100
            df_sectors_all[(index[0], '% per Workforce')] = row/df_sectors_all.loc[df_sectors_all[df_sectors_all[('Industry class / branch (SIC2008)', 'Sector Name')] == 'Total (excluding A-U)'].index.values.astype(int)[0], ('Total Workforce', 'n')]#*100
        if ('Total' in index[0]):
            df_sectors_all[(index[0], '% Sector per Workforce')] = row/df_sectors_all.loc[df_sectors_all[df_sectors_all[('Industry class / branch (SIC2008)', 'Sector Name')] == 'Total (excluding A-U)'].index.values.astype(int)[0], ('Total Workforce', 'n')]#*100

    # Set cut-off
    # Gender
    total_female = df_sectors_all.loc[df_sectors_all[df_sectors_all[('Industry class / branch (SIC2008)', 'Sector Name')] == 'Total (excluding A-U)'].index.values.astype(int)[0], ('Female', '% per Workforce')]
    female_dominated = total_female + (int(gender_ratio) / 100)
    df_sectors_all.loc[df_sectors_all[('Female', '% per Sector')] >= female_dominated, ('Sectoral Gender Segregation', 'Dominant Category')] = 'Female'
    male_dominated = total_female - (int(gender_ratio) / 100)
    df_sectors_all.loc[df_sectors_all[('Female', '% per Sector')] <= male_dominated, ('Sectoral Gender Segregation', 'Dominant Category')] = 'Male'
    df_sectors_all.loc[(df_sectors_all[('Female', '% per Sector')] > male_dominated) & (df_sectors_all[('Female', '% per Sector')] < female_dominated) & (df_sectors_all[('Industry class / branch (SIC2008)', 'Sector Name')].astype(str) != 'Total (excluding A-U)'), ('Sectoral Gender Segregation', 'Dominant Category')] = 'Mixed Gender'
    # Age
    total_old = df_sectors_all.loc[df_sectors_all[df_sectors_all[('Industry class / branch (SIC2008)', 'Sector Name')] == 'Total (excluding A-U)'].index.values.astype(int)[0], (f'Older (>= {age_limit} years)', '% per Workforce')]
    old_dominated = total_old + (int(age_ratio) / 100)
    df_sectors_all.loc[df_sectors_all[(f'Older (>= {age_limit} years)', '% per Sector')] >= old_dominated, ('Sectoral Age Segregation', 'Dominant Category')] = 'Older'
    young_dominated = total_old - (int(age_ratio) / 100)
    df_sectors_all.loc[df_sectors_all[(f'Older (>= {age_limit} years)', '% per Sector')] <= young_dominated, ('Sectoral Age Segregation', 'Dominant Category')] = 'Younger'
    df_sectors_all.loc[(df_sectors_all[(f'Older (>= {age_limit} years)', '% per Sector')] < old_dominated) & (df_sectors_all[(f'Older (>= {age_limit} years)', '% per Sector')] > young_dominated) & (df_sectors_all[('Industry class / branch (SIC2008)', 'Sector Name')].astype(str) != 'Total (excluding A-U)'), ('Sectoral Age Segregation', 'Dominant Category')] = 'Mixed Age'

    # Add AU and other rows
    au.insert(2, 'Keywords', np.nan)
    au.insert(3, 'Keywords Count', np.nan)
    au[['Sectoral Gender Segregation', 'Sectoral Age Segregation']] = np.nan
    au.columns = pd.MultiIndex.from_tuples([col for col in df_sectors_all.columns if '%' not in col[1]])
    df_sectors_all = pd.concat([au, df_sectors_all], ignore_index=True)

    # Arrange columns
    df_sectors_all = df_sectors_all.reindex(columns=df_sectors_all.columns.reindex(['Industry class / branch (SIC2008)', 'Female', 'Male', 'Sectoral Gender Segregation', f'Older (>= {age_limit} years)', f'Younger (< {age_limit} years)', 'Sectoral Age Segregation', 'Total Workforce'], level=0)[0])
    df_sectors_all = df_sectors_all.reindex(columns=df_sectors_all.columns.reindex(['Code', 'Sector Name', 'Keywords', 'Keywords Count', 'n', '% per Sector', '% per Social Category', '% per Workforce', '% Sector per Workforce', 'Dominant Category'], level=1)[0])

    level1_cols_tuple = []
    for col in df_sectors_all.columns:
        if ('SIC2008' in col[0]):
            level1_cols_tuple.append(('SBI Sector Titles', *col))
        elif (re.search(r'[Mm]ale', col[0])) or ('Gender' in col[0]):
            level1_cols_tuple.append(('Gender', *col))
        elif ('45' in col[0]) or ('Age' in col[0]):
            level1_cols_tuple.append(('Age', *col))
        elif ('Total' in col[0]):
            level1_cols_tuple.append(('Total Workforce', *col))

    df_sectors_all.columns = pd.MultiIndex.from_tuples(level1_cols_tuple, names=['Variables', 'Categories', 'Counts'])

    if save_enabled is True:
        df_sectors_all.to_csv(f'{data_save_dir}Sectors Output from script.csv', index=False)
        df_sectors_all.to_pickle(f'{data_save_dir}Sectors Output from script.pkl')
        with pd.option_context('max_colwidth', 10000000000):
            df_sectors_all.style.to_latex(f'{data_save_dir}Sectors Output from script.tex', index=False, longtable=True, escape=True, multicolumn=True, multicolumn_format='c', position='H', caption='Sectoral Gender and Age Composition and Segregation, Keywords, Counts, and Percentages', label='Jobs Count per Sector (x 1000)')
        df_sectors_all.to_markdown(f'{data_save_dir}Sectors Output from script.md', index=True)
        save_sector_excel(df_sectors_all, data_save_dir)

    return df_sectors_all


# %%
# Function to read and save keyword lists
def read_and_save_keyword_list(
    print_enabled: bool = False,
    save_enabled: bool = True,
    translate_enabled: bool = False,
    sectors_file_path: str = validate_path(f'{code_dir}/scraped_data/CBS/Found Data/'),
    use_top10_data: bool = False,
    age_limit: int = 45,
    age_ratio: int = 10,
    gender_ratio: int = 20,
):

    if print_enabled is True:
        print(
            f'NOTE: The function "read_and_save_keyword_list" contains the following optional (default) arguments:\n{get_default_args(read_and_save_keyword_list)}'
        )
    # Augment Keywords List
    # Gender
    if use_top10_data is True:
        # Highest % of women per occupation
        keyword_file_path_womenvocc = validate_path(
            f'{sectors_file_path}Top 10 highest % of women in occupations (2018).csv'
        )
        df_womenvocc = pd.read_csv(keyword_file_path_womenvocc)
        #
        _keywords_womenvocc = df_womenvocc['Beroep'].loc[1:].to_list()

        # Highest % of men per occupation
        keyword_file_path_menvocc = validate_path(
            f'{sectors_file_path}Top 10 highest % of men in occupations (2018).csv'
        )
        df_menvocc = pd.read_csv(keyword_file_path_menvocc)
        keywords_menvocc = df_menvocc['Beroep'].loc[1:].to_list()
    elif use_top10_data is False:
        keywords_womenvocc = []
        keywords_menvocc = []

    # Read into df
    df_sectors = get_sector_df_from_cbs()
    df_sectors.set_index(('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name'), inplace = True)

    # Gender Sectors DFs
    df_sector_gen_mixed = df_sectors.loc[df_sectors[('Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Mixed Gender']
    df_sector_women = df_sectors.loc[df_sectors[('Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Female']
    df_sector_men = df_sectors.loc[df_sectors[('Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Male']

    # Make Mixed Gender keywords list
    keywords_genvsect = df_sector_gen_mixed.index.to_list()
    keywords_genvsect = clean_and_translate_keyword_list(keywords_genvsect, translate_enabled)

    # Add female and male sectors to lists
    # Female Sectors + DF women v occ
    for keywords_list in df_sector_women[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].to_list():
        keywords_womenvocc.extend(keywords_list)
        keywords_womenvocc.extend(df_sector_men.index.to_list())
    keywords_womenvocc = clean_and_translate_keyword_list(keywords_womenvocc, translate_enabled)

    # Male Sectors + DF men v occ
    for keywords_list in df_sector_men[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].to_list():
        keywords_menvocc.extend(keywords_list)
        keywords_menvocc.extend(df_sector_men.index.to_list())
    keywords_menvocc = clean_and_translate_keyword_list(keywords_menvocc, translate_enabled)

    ################################################### AGE ###################################################
    # Age Sectors DFs
    df_sector_age_mixed = df_sectors.loc[df_sectors[('Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Mixed Age']
    df_sector_old = df_sectors.loc[df_sectors[('Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Older']
    df_sector_young = df_sectors.loc[df_sectors[('Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Younger']

    # Make Mixed Age keywords list
    keywords_agevsect = df_sector_age_mixed.index.to_list()
    keywords_agevsect = clean_and_translate_keyword_list(keywords_agevsect, translate_enabled)

    # Add older and younger sectors to lists
    # Older Sectors
    keywords_oldvocc = df_sector_old.index.to_list()
    keywords_oldvocc = clean_and_translate_keyword_list(keywords_oldvocc, translate_enabled)

    # Younger Sectors
    keywords_youngvocc = df_sector_young.index.to_list()
    keywords_youngvocc = clean_and_translate_keyword_list(keywords_youngvocc, translate_enabled)

    ################################################### SAVE ###################################################

    # Print and save lists
    if print_enabled is True:
        print(f'Female keywords total {len(keywords_womenvocc)}:\n{keywords_womenvocc}\n')
        print(f'Male keywords total {len(keywords_menvocc)}:\n{keywords_menvocc}\n')
        print(
            f'Mixed gender keywords total {len(keywords_genvsect)}:\n{keywords_genvsect}\n'
        )
        print(f'Older worker keywords total {len(keywords_oldvocc)}:\n{keywords_oldvocc}\n')
        print(
            f'Younger keywords total {len(keywords_youngvocc)}:\n{keywords_youngvocc}\n'
        )
        print(
            f'Mixed age keywords total {len(keywords_agevsect)}:\n{keywords_agevsect}\n'
        )

    keywords_dict = {
        'keywords_womenvocc': keywords_womenvocc,
        'keywords_menvocc': keywords_menvocc,
        'keywords_genvsect': keywords_genvsect,
        'keywords_oldvocc': keywords_oldvocc,
        'keywords_youngvocc': keywords_youngvocc,
        'keywords_agevsect': keywords_agevsect,
    }
    if save_enabled is True:
        with open(
            f'{code_dir}/scraped_data/CBS/Data/keywords_dict.json', 'w', encoding='utf8'
        ) as f:
            json.dump(keywords_dict, f)

        for key, value in keywords_dict.items():
            if translate_enabled is False:
                save_path_file_name = f'Sectors List/{str(key)}.txt'
            elif translate_enabled is True:
                save_path_file_name = (
                    f'Sectors List/{str(key)}_with_nl.txt'
                )

            if print_enabled is True:
                print(
                    f'Saving {key} of length: {len(value)} to file location {sectors_file_path}.'
                )
            with open(sectors_file_path + save_path_file_name, 'w') as f:
                for i in value:
                    f.write(f'{i.lower()}\n')

    elif save_enabled is False:
        print('No keyword list save enabled.')

    return (
        keywords_dict,
        keywords_womenvocc,
        keywords_menvocc,
        keywords_genvsect,
        keywords_oldvocc,
        keywords_youngvocc,
        keywords_agevsect,
        df_sector_women,
        df_sector_men,
        df_sector_old,
        df_sector_young,
    )


# %%
def get_keywords_from_cbs(
    save_enabled: bool = True,
    keywords_file_path: str = f'{code_dir}/scraped_data/CBS/Found Data/Sectors List/',
    cols = ['Industry class / branch (SIC2008)', 'Sex of employee', 'Other characteristics employee', 'Employment/Jobs (x 1 000)'],
    age_limit: int = 45,
    age_ratio: int = 10,
    gender_ratio: int = 20,
):

    keywords_dict, keywords_womenvocc, keywords_menvocc, keywords_genvsect, keywords_oldvocc, keywords_youngvocc, keywords_agevsect, df_sector_women, df_sector_men, df_sector_old, df_sector_young = read_and_save_keyword_list()

    df_sectors = get_sector_df_from_cbs()
    df_sectors.set_index(('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name'), inplace = True)

    # Make dfs, lists and dicts for each group
    sectors_list = clean_and_translate_keyword_list(df_sectors.loc['Agriculture and industry': 'Other service activities', ('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].index.to_list())

    female_sectors = df_sectors.loc[df_sectors[('Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Female']

    female_list = clean_and_translate_keyword_list(female_sectors.index.to_list())
    female_dict = female_sectors.to_dict('index')

    male_sectors = df_sectors.loc[df_sectors[('Gender', 'Sectoral Gender Segregation', 'Dominant Category')] == 'Male']
    male_list = clean_and_translate_keyword_list(male_sectors.index.to_list())
    male_dict = male_sectors.to_dict()

    all_gender_sectors = pd.concat([female_sectors, male_sectors])
    all_gender_list = clean_and_translate_keyword_list(all_gender_sectors.index.to_list())
    all_gender_dict = all_gender_sectors.to_dict()

    old_sectors = df_sectors.loc[df_sectors[('Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Older']
    old_list = clean_and_translate_keyword_list(old_sectors.index.to_list())
    old_dict = old_sectors.to_dict()

    young_sectors = df_sectors.loc[df_sectors[('Age', 'Sectoral Age Segregation', 'Dominant Category')] == 'Younger']
    young_list = clean_and_translate_keyword_list(young_sectors.index.to_list())
    young_dict = young_sectors.to_dict()

    all_age_sectors = pd.concat([old_sectors, young_sectors])
    all_age_list = clean_and_translate_keyword_list(all_age_sectors.index.to_list())
    all_age_dict = all_age_sectors.to_dict()

    # Save lists
    if save_enabled is True:
        with open(f'{keywords_file_path}keywords_sectors_FROM_SECTOR.txt', 'w') as f:
            for i in sectors_list:
                f.write(f'{i.lower()}\n')

        with open(
            f'{keywords_file_path}keywords_womenvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
            'w',
        ) as f:
            for i in female_list:
                f.write(f'{i.lower()}\n')

        with open(
            f'{keywords_file_path}keywords_menvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
            'w',
        ) as f:
            for i in male_list:
                f.write(f'{i.lower()}\n')

        with open(
            f'{keywords_file_path}keywords_genvsect_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
            'w',
        ) as f:
            for i in all_gender_list:
                f.write(f'{i.lower()}\n')

        with open(
            f'{keywords_file_path}keywords_oldvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
            'w',
        ) as f:
            for i in old_list:
                f.write(f'{i.lower()}\n')

        with open(
            f'{keywords_file_path}keywords_youngvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
            'w',
        ) as f:
            for i in young_list:
                f.write(f'{i.lower()}\n')

        with open(
            f'{keywords_file_path}keywords_agevsect_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
            'w',
        ) as f:
            for i in all_age_list:
                f.write(f'{i.lower()}\n')

    return (
        df_sectors,
        female_sectors,
        male_sectors,
        all_gender_sectors,
        old_sectors,
        young_sectors,
        all_age_sectors,
    )


# %%
# Find file location
def get_keyword_list(
    print_enabled: bool = False,
    get_from_cbs: bool = True,
    age_limit=45,
    age_ratio=10,
    gender_ratio=20,
):

    keywords_file_path = validate_path(
        f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Analysis and Dataset Used/'
    )

    if get_from_cbs is True:
        (
            df_sectors,
            female_sectors,
            male_sectors,
            all_gender_sectors,
            old_sectors,
            young_sectors,
            all_age_sectors,
        ) = get_keywords_from_cbs(
            save_enabled=True,
            age_limit=age_limit,
            age_ratio=age_ratio,
            gender_ratio=gender_ratio,
        )

    # Women Sector
    keywords_dict, keywords_womenvocc, keywords_menvocc, keywords_genvsect, keywords_oldvocc, keywords_youngvocc, keywords_agevsect, df_sector_women, df_sector_men, df_sector_old, df_sector_young = read_and_save_keyword_list()

    with open(keywords_file_path + 'keywords_womenvocc.txt', 'r') as f:
        keywords_womenvocc = f.read().splitlines()
    with open(
        keywords_file_path
        + f'keywords_womenvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
        'r',
    ) as f:
        keywords_womenvocc_sectors = f.read().splitlines()
        keywords_womenvocc.extend(keywords_womenvocc_sectors)
    if 'busines' in keywords_womenvocc:
        keywords_womenvocc.remove('busines')
    if 'busine' in keywords_womenvocc:
        keywords_womenvocc.remove('busine')
    keywords_womenvocc = list(filter(None, list(set(keywords_womenvocc))))
    keywords_womenvocc = clean_and_translate_keyword_list(keywords_womenvocc)

    # Men Sector
    with open(keywords_file_path + 'keywords_menvocc.txt', 'r') as f:
        keywords_menvocc = f.read().splitlines()
    with open(
        keywords_file_path
        + f'keywords_menvocc_ratio-{str(gender_ratio)}_FROM_SECTOR.txt',
        'r',
    ) as f:
        keywords_menvocc_sectors = f.read().splitlines()
        keywords_menvocc.extend(keywords_menvocc_sectors)
    if 'busines' in keywords_menvocc:
        keywords_menvocc.remove('busines')
    if 'busine' in keywords_menvocc:
        keywords_menvocc.remove('busine')
    keywords_menvocc = list(filter(None, list(set(keywords_menvocc))))
    keywords_menvocc = clean_and_translate_keyword_list(keywords_menvocc)

    # Gender Segregated Sector
    keywords_genvsect = keywords_womenvocc + keywords_menvocc
    if 'busines' in keywords_genvsect:
        keywords_genvsect.remove('busines')
    if 'busine' in keywords_genvsect:
        keywords_genvsect.remove('busine')
    keywords_genvsect = list(filter(None, list(set(keywords_genvsect))))
    keywords_genvsect = clean_and_translate_keyword_list(keywords_genvsect)

    # Old worker Sector
    with open(keywords_file_path + 'keywords_oldvocc.txt', 'r') as f:
        keywords_oldvocc = f.read().splitlines()
    with open(
        keywords_file_path
        + f'keywords_oldvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
        'r',
    ) as f:
        keywords_oldvocc_sectors = f.read().splitlines()
        keywords_oldvocc.extend(keywords_oldvocc_sectors)
    if 'busines' in keywords_oldvocc:
        keywords_oldvocc.remove('busines')
    if 'busine' in keywords_oldvocc:
        keywords_oldvocc.remove('busine')
    keywords_oldvocc = list(filter(None, list(set(keywords_oldvocc))))
    keywords_oldvocc = clean_and_translate_keyword_list(keywords_oldvocc)

    # Young worker Sector
    with open(keywords_file_path + 'keywords_youngvocc.txt', 'r') as f:
        keywords_youngvocc = f.read().splitlines()
    with open(
        keywords_file_path
        + f'keywords_youngvocc_{str(age_limit)}_ratio-{str(age_ratio)}_FROM_SECTOR.txt',
        'r',
    ) as f:
        keywords_youngvocc_sectors = f.read().splitlines()
        keywords_youngvocc.extend(keywords_youngvocc_sectors)
    if 'busines' in keywords_youngvocc:
        keywords_youngvocc.remove('busines')
    if 'busine' in keywords_youngvocc:
        keywords_youngvocc.remove('busine')
    keywords_youngvocc = list(filter(None, list(set(keywords_youngvocc))))
    keywords_youngvocc = clean_and_translate_keyword_list(keywords_youngvocc)

    # Age Segregated Sector
    keywords_agevsect = keywords_oldvocc + keywords_youngvocc
    if 'busines' in keywords_agevsect:
        keywords_agevsect.remove('busines')
    if 'busine' in keywords_agevsect:
        keywords_agevsect.remove('busine')
    keywords_agevsect = list(filter(None, list(set(keywords_agevsect))))
    keywords_agevsect = clean_and_translate_keyword_list(keywords_agevsect)

    # All Sector
    sbi_english_keyword_list, sbi_english_keyword_dict, sbi_sectors_dict, sbi_sectors_dict_full, sbi_sectors_dom_gen, sbi_sectors_dom_age, sbi_sectors_keywords_gen_dom, sbi_sectors_keywords_age_dom, sbi_sectors_keywords_full_dom, trans_keyword_list = get_sbi_sectors_list()
    # keywords_sector = list(set([y for x in df_sectors.loc['Agriculture and industry': 'Other service activities', ('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')].values.tolist() if isinstance(x, list) for y in x]))
    keywords_sector = trans_keyword_list
    with open(keywords_file_path + 'keywords_sectors_FROM_SECTOR.txt', 'r') as f:
        keywords_sector_sectors = f.read().splitlines()
    keywords_sector.extend(keywords_sector_sectors)

    if 'busines' in keywords_sector:
        keywords_sector.remove('busines')
    if 'busine' in keywords_sector:
        keywords_sector.remove('busine')
    keywords_sector = list(filter(None, list(set(keywords_sector))))
    keywords_sector = clean_and_translate_keyword_list(keywords_sector)

    with open(keywords_file_path + 'keywords_sector.txt', 'w') as f:
        for word in keywords_sector:
            f.write(word + '\n')

    keywords_list = (
        keywords_sector
        + keywords_womenvocc
        + keywords_menvocc
        + keywords_oldvocc
        + keywords_youngvocc
    )
    if 'busines' in keywords_list:
        keywords_list.remove('busines')
    if 'busine' in keywords_list:
        keywords_list.remove('busine')
    # Remove duplicates
    keywords_list = list(filter(None, list(set(keywords_list))))
    keywords_list = clean_and_translate_keyword_list(keywords_list)

    # Add mixed gender
    # mixed_gender = [x for x in keywords_list if not x in keywords_womenvocc]
    # mixed_gender = [x for x in mixed_gender if not x in keywords_menvocc]
    # keywords_genvsect.extend(mixed_gender)
    mixed_gender = [
        k
        for k in keywords_list
        if (k not in keywords_womenvocc) and (k not in keywords_menvocc)
    ]
    if 'busines' in mixed_gender:
        mixed_gender.remove('busines')
    if 'busine' in mixed_gender:
        mixed_gender.remove('busine')
    mixed_gender = list(filter(None, list(set(mixed_gender))))
    mixed_gender = clean_and_translate_keyword_list(mixed_gender)
    mixed_age = [
        k
        for k in keywords_list
        if (k not in keywords_oldvocc) and (k not in keywords_youngvocc)
    ]
    if 'busines' in mixed_age:
        mixed_age.remove('busines')
    if 'busine' in mixed_age:
        mixed_age.remove('busine')
    mixed_age = list(filter(None, list(set(mixed_age))))
    mixed_age = clean_and_translate_keyword_list(mixed_age)

    if print_enabled is True:
        # Print and save lists
        print(f'All sector total {len(keywords_sector)}:\n{keywords_sector}\n')
        print(
            f'Female keywords total {len(keywords_womenvocc)}:\n{keywords_womenvocc}\n'
        )
        print(f'Male keywords total {len(keywords_menvocc)}:\n{keywords_menvocc}\n')
        print(
            f'Gender Segregated total {len(keywords_genvsect)}:\n{keywords_genvsect}\n'
        )
        print(f'Mixed Gender total {len(mixed_gender)}:\n{mixed_gender}\n')
        print(
            f'Older worker keywords total {len(keywords_oldvocc)}:\n{keywords_oldvocc}\n'
        )
        print(
            f'Younger keywords total {len(keywords_youngvocc)}:\n{keywords_youngvocc}\n'
        )
        print(f'Age Segregated total {len(keywords_agevsect)}:\n{keywords_agevsect}\n')
        print(f'Mixed Age total {len(mixed_age)}:\n{mixed_age}\n')
        print(f'All Keywords total {len(keywords_list)}:\n{keywords_list}')

    return (
        keywords_list,
        keywords_sector,
        keywords_womenvocc,
        keywords_menvocc,
        keywords_genvsect,
        keywords_oldvocc,
        keywords_youngvocc,
        keywords_agevsect,
    )

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
    file_save_format = 'pkl',
    file_save_format_backup = 'csv',
    image_save_format = 'eps',
):
    parent_dir = validate_path(f'{data_save_path}content analysis + ids + sectors/')
    content_analysis_dir = validate_path(f'{scraped_data}/Coding Material/')
    df_dir = validate_path(f'{data_save_path}final dfs/')
    models_save_path=validate_path(f'{data_save_path}classification models/')
    table_save_path=validate_path(f'{data_save_path}output tables/')
    plot_save_path=validate_path(f'{data_save_path}plots/')
    embeddings_save_path=validate_path(f'{data_save_path}embeddings models/')

    (
        keywords_list,
        keywords_sector,
        keywords_womenvocc,
        keywords_menvocc,
        keywords_genvsect,
        keywords_oldvocc,
        keywords_youngvocc,
        keywords_agevsect,
    ) = get_keyword_list()

    (
        sbi_english_keyword_list,
        sbi_english_keyword_dict,
        sbi_sectors_dict,
        sbi_sectors_dict_full,
        sbi_sectors_dom_gen,
        sbi_sectors_dom_age,
        sbi_sectors_keywords_gen_dom,
        sbi_sectors_keywords_age_dom,
        sbi_sectors_keywords_full_dom,
        trans_keyword_list
    ) = get_sbi_sectors_list()

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
        'keywords_list': keywords_list,
        'keywords_sector': keywords_sector,
        'keywords_womenvocc': keywords_womenvocc,
        'keywords_menvocc': keywords_menvocc,
        'keywords_genvsect': keywords_genvsect,
        'keywords_oldvocc': keywords_oldvocc,
        'keywords_youngvocc': keywords_youngvocc,
        'keywords_agevsect': keywords_agevsect,
        'sbi_english_keyword_list': sbi_english_keyword_list,
        'sbi_english_keyword_dict': sbi_english_keyword_dict,
        'sbi_sectors_dict': sbi_sectors_dict,
        'sbi_sectors_dict_full': sbi_sectors_dict_full,
        'sbi_sectors_dom_gen': sbi_sectors_dom_gen,
        'sbi_sectors_dom_age': sbi_sectors_dom_age,
        'trans_keyword_list': trans_keyword_list,
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
    json_file_name = f'{site.lower()}_jobs_dict_{keyword_file.lower()}.json'.replace("-Noon's MacBook Pro", '').replace('_(excl', '')
    df_file_name = f'{site.lower()}_jobs_df_{keyword_file.lower()}.{args["file_save_format_backup"]}'.replace("-Noon's MacBook Pro", '').replace('_(excl', '')
    logs_file_name = f'{site.lower()}_jobs_logs_{keyword_file.lower()}.log'.replace("-Noon's MacBook Pro", '').replace('_(excl', '')
    filemode = 'a+' if is_non_zero_file(save_path + logs_file_name.lower()) is True else 'w+'

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
            print(f'Unexpected error with stale element checker: {sys.exc_info()[0]}.')


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
            print(f'Unexpected error with act cool checker: {sys.exc_info()[0]}.')


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

        previous = itertools.islice(parent.children, 0, parent.contents.index(child))
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
        type_check = np.all([isinstance(val, str) for val in df_old_jobs['Job ID']])
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

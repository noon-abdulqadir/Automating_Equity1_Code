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
def save_trans_keyword_list(trans_keyword_list, parent_dir=validate_path(f'{code_dir}/data/content analysis + ids + sectors/')):

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


# %%
# Function to get translated and cleaned keyword list
def get_trans_keyword_list(parent_dir=validate_path(f'{code_dir}/data/content analysis + ids + sectors/')):

    with open(f'{parent_dir}trans_keyword_list.txt', 'r') as f:
        trans_keyword_list = [line.rstrip(' \n') for line in f]

    trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

    return trans_keyword_list

# %% Function to get sbi_sectors_dict
def get_sbi_sectors_list(
    save_enabled=True,
    parent_dir=validate_path(f'{code_dir}/data/content analysis + ids + sectors/'),
    ):

    sib_5_loc = validate_path(f'{parent_dir}Sectors + Age and Gender Composition of Industires and Jobs/Found Data for Specific Occupations/SBI_ALL_NACE_REV2.csv')
    trans_keyword_save_loc=f'{parent_dir}Sectors + Age and Gender Composition of Industires and Jobs/Output CSVs for Occupational Sectors/'
    trans_keyword_list = get_trans_keyword_list()

    df_sbi_sectors = pd.read_csv(sib_5_loc, delimiter=',')
    df_sbi_sectors.columns = df_sbi_sectors.columns.str.strip()
    df_sbi_sectors.rename(columns = {'Description': 'Old_Sector_Name'}, inplace=True)
    df_sbi_sectors = df_sbi_sectors.dropna(subset=['Old_Sector_Name', 'Code'])
    df_sbi_sectors['Old_Sector_Name'] = df_sbi_sectors['Old_Sector_Name'].apply(lambda x: x.lower().strip())
    df_sbi_sectors = df_sbi_sectors.loc[df_sbi_sectors['Level'] == 1]
    df_sbi_sectors.drop(columns=['Level', 'Parent', 'This item includes', 'This item also includes', 'Rulings', 'This item excludes', 'Reference to ISIC Rev. 4'], inplace=True)

    df_sectors_all = pd.read_pickle(f'{trans_keyword_save_loc}Sectors Output from script.pkl')[[('SBI Sector Titles'), ('Gender'), ('Age')]].droplevel('Categories', axis=1)[[('SBI Sector Titles', 'Code'), ('SBI Sector Titles', 'Sector Name'), ('SBI Sector Titles', 'Keywords'), ('Gender', 'Dominant Category'), ('Age', 'Dominant Category')]].droplevel('Variables', axis=1)
    df_sectors_all.columns = ['Code', 'Sector Name', 'Keywords', 'Gender Dominant Category', 'Age Dominant Category']
    df_sbi_sectors = df_sbi_sectors.merge(df_sectors_all, how='inner', on='Code')
    df_sbi_sectors.rename(columns = {'Sector Name': 'Sector_Name', 'Keywords': 'Used_Sector_Keywords', 'Gender Dominant Category': 'Gender_Dominant_Category', 'Age Dominant Category': 'Age_Dominant_Category'}, inplace=True)
    df_sbi_sectors['Sector_Name'] = df_sbi_sectors['Sector_Name'].apply(lambda x: x.strip().lower() if isinstance(x, str) else np.nan)
    # df_sbi_sectors['Gender_Dominant_Category'] = df_sbi_sectors['Gender_Dominant_Category'].apply(lambda x: x.strip().lower() if isinstance(x, str) else np.nan)
    # df_sbi_sectors['Age_Dominant_Category'] = df_sbi_sectors['Age_Dominant_Category'].apply(lambda x: x.strip().lower() if isinstance(x, str) else np.nan)
    df_sbi_sectors['Used_Sector_Keywords'] = df_sbi_sectors['Used_Sector_Keywords'].apply(lambda x: clean_and_translate_keyword_list(x) if isinstance(x, list) else np.nan)
    df_sbi_sectors.set_index(df_sbi_sectors['Code'], inplace=True)

    df_sbi_sectors.to_csv(f'{trans_keyword_save_loc}SBI-5_Sectors.csv', index=True)
    df_sbi_sectors.to_excel(f'{trans_keyword_save_loc}SBI-5_Sectors.xlsx', index=True)
    df_sbi_sectors.to_pickle(f'{trans_keyword_save_loc}SBI-5_Sectors.pkl')

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
    for index, row in df_sbi_sectors.iterrows():
        sbi_sectors_dict_full[row['Sector_Name']] = row['Used_Sector_Keywords']
        sbi_sectors_dom_gen[row['Sector_Name']] = row['Gender_Dominant_Category']
        sbi_sectors_dom_age[row['Sector_Name']] = row['Age_Dominant_Category']

    if save_enabled is True:
        with open(f'{parent_dir}sbi_english_keyword_list.txt', 'w', encoding='utf8') as f:
            for i in sbi_english_keyword_list:
                f.write(f'{i.lower()}\n')
        with open(f'{parent_dir}sbi_english_keyword_dict.json', 'w', encoding='utf8') as f:
            json.dump(sbi_english_keyword_dict, f)
        with open(f'{parent_dir}sbi_sectors_dict.json', 'w', encoding='utf8') as f:
            json.dump(sbi_sectors_dict, f)

    return sbi_english_keyword_list, sbi_english_keyword_dict, sbi_sectors_dict, sbi_sectors_dict_full, sbi_sectors_dom_gen, sbi_sectors_dom_age, trans_keyword_list


# %% CBS Data request
def get_cbs_odata(
    tables_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Output CSVs for Occupational Sectors/',
    sectors_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Found Data for Specific Occupations/',
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

    with open(f'{tables_file_path}cbs_data_info.json', 'w', encoding='utf8') as f:
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
    with open(f'{tables_file_path}cbs_data_dimensions.json', 'w', encoding='utf8') as f:
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
    keywords_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Analysis and Dataset Used/',
    sectors_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Found Data for Specific Occupations/',
    tables_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Output CSVs for Occupational Sectors/',
    cols = ['Industry class / branch (SIC2008)', 'Sex of employee', 'Other characteristics employee', 'Employment/Jobs (x 1 000)'],
    get_cbs_odata_enabled=False,
    age_limit: int = 45,
    age_ratio: int = 10,
    gender_ratio: int = 20,
):
    # with open(f'{code_dir}/data/content analysis + ids + sectors/sbi_sectors_dict.json', 'r', encoding='utf8') as f:
    #     sbi_sectors_dict = json.load(f)
    sbi_english_keyword_list, sbi_english_keyword_dict, sbi_sectors_dict, sbi_sectors_dict_full, sbi_sectors_dom_gen, sbi_sectors_dom_age, trans_keyword_list = get_sbi_sectors_list()

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
        df_sectors_all.to_csv(f'{tables_file_path}Sectors Output from script.csv', index=False)
        df_sectors_all.to_pickle(f'{tables_file_path}Sectors Output from script.pkl')
        with pd.option_context("max_colwidth", 10000000000):
            df_sectors_all.to_latex(f'{tables_file_path}Sectors Output from script.tex', index=False, longtable=True, escape=True, multicolumn=True, multicolumn_format='c', position='H', caption='Sectoral Gender and Age Composition and Segregation, Keywords, Counts, and Percentages', label='Jobs Count per Sector (x 1000)')
        df_sectors_all.to_markdown(f'{tables_file_path}Sectors Output from script.md', index=True)
        save_sector_excel(df_sectors_all, tables_file_path)

    return df_sectors_all


# %%
# Function to read and save keyword lists
def read_and_save_keyword_list(
    print_enabled: bool = False,
    save_enabled: bool = True,
    translate_enabled: bool = False,
    sectors_file_path: str = validate_path(f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/'),
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
            f'{sectors_file_path}Found Data for Specific Occupations/Top 10 highest % of women in occupations (2018).csv'
        )
        df_womenvocc = pd.read_csv(keyword_file_path_womenvocc)
        #
        _keywords_womenvocc = df_womenvocc['Beroep'].loc[1:].to_list()

        # Highest % of men per occupation
        keyword_file_path_menvocc = validate_path(
            f'{sectors_file_path}Found Data for Specific Occupations/Top 10 highest % of men in occupations (2018).csv'
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
            f'Younger worker keywords total {len(keywords_youngvocc)}:\n{keywords_youngvocc}\n'
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
            f'{sectors_file_path}/Analysis and Dataset Used/keywords_dict.json', 'w', encoding='utf8'
        ) as f:
            json.dump(keywords_dict, f)

        for key, value in keywords_dict.items():
            if translate_enabled is False:
                save_path_file_name = f'/Analysis and Dataset Used/{str(key)}.txt'
            elif translate_enabled is True:
                save_path_file_name = (
                    f'/Analysis and Dataset Used/{str(key)}_with_nl.txt'
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
    keywords_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Analysis and Dataset Used/',
    sectors_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Found Data for Specific Occupations/',
    tables_file_path: str = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Output CSVs for Occupational Sectors/',
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
    sbi_english_keyword_list, sbi_english_keyword_dict, sbi_sectors_dict, sbi_sectors_dict_full, sbi_sectors_dom_gen, sbi_sectors_dom_age, trans_keyword_list = get_sbi_sectors_list()
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
            f'Younger worker keywords total {len(keywords_youngvocc)}:\n{keywords_youngvocc}\n'
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
    age_sectors=['Older Worker', 'Younger Worker', 'Mixed Age'],
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
    content_analysis_dir = validate_path(f'{parent_dir}Coding Material/')
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
        trans_keyword_list,
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
    json_file_name = f'{site.lower()}_jobs_dict_{keyword_file.lower()}.json'.replace("-Noon's MacBook Pro", '').replace("_(excl", '')
    df_file_name = f'{site.lower()}_jobs_df_{keyword_file.lower()}.{args["file_save_format_backup"]}'.replace("-Noon's MacBook Pro", '').replace("_(excl", '')
    logs_file_name = f'{site.lower()}_jobs_logs_{keyword_file.lower()}.log'.replace("-Noon's MacBook Pro", '').replace("_(excl", '')
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
# load_clean_save_dicts_dfs

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


# %% [markdown]
# post_collection_processing
# %%
# Function to create a metadata dict
def create_metadata(args=get_args()):

    if args['print_enabled'] is True:
        print('Creating Metadata Dict.')

    columns = args['columns_fill_list'] + args['columns_drop_list']

    metadata_dict = {
        columns[0]: {
            'type': 'str',
            'description': 'unique identifier for each job ad',
            'example': 'pj_245d9fed724ade0b',
        },  # Job ID
        'Language': {
            'type': 'str',
            'description': 'language of job ad',
            'example': 'en',
        },  # Language
        'Job Description': {
            'type': 'str',
            'description': 'job posting complete text',
        },  # Job Description
        # Sentence
        columns[1]: {'type': 'str', 'description': 'sentence from job ad'},
        columns[2]: {
            'type': 'str',
            'description': 'search term used for data collection',
            'list': args['keywords_list'],
        },  # Search Keyword
        columns[3]: {
            'type': 'str',
            'description': 'gender categorization of job based on Standard Industrial Classifications (SBI2008; 2018)',
            'example': 'Female',
        },  # Gender
        columns[4]: {
            'type': 'str',
            'description': 'age categorization of job based on Standard Industrial Classifications (SBI2008; 2018)',
            'example': 'Older Worker',
        },  # Age
        columns[5]: {
            'type': 'str',
            'description': 'name of job search online platforms from which data was collected',
            'list': args['site_list'],
        },  # Platform
        columns[6]: {
            'type': 'str',
            'description': 'title of advertised job position',
            'example': 'Store Assistant',
        },  # Job Title
        columns[7]: {
            'type': 'str',
            'description': 'name of company that posted job ad',
            'example': 'Picnic',
        },  # Company Name
        columns[8]: {
            'type': 'str',
            'description': 'city of advertised job position',
            'example': 'Amstelveen',
        },  # Location
        columns[9]: {
            'type': 'float',
            'description': 'ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - rating of company out of 5 stars',
            'example': 4.7,
        },  # Rating
        columns[10]: {
            'type': ['str', 'list'],
            'description': 'industry of advertised job position',
            'example': ['Research', 'Chemicals', 'Food Production'],
        },  # Industry
        columns[11]: {
            'type': ['str', 'list'],
            'description': 'sector of advertised job position',
            'example': ['Engineering', 'Information Technology'],
        },  # Sector
        columns[12]: {
            'type': 'str',
            'description': 'IN DUTCH - ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - type of company ownership',
            'example': 'Beursgenoteerd bedrijf',
        },  # Type of ownership
        columns[13]: {
            'type': 'str',
            'description': 'IN DUTCH - ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - type of employment for advertised job position',
        },  # Employment Type
        columns[14]: {
            'type': 'str',
            'description': 'IN DUTCH - ONLY AVAILABLE FOR GLASSDOOR PLATFORM JOB ADS - level of seniority for advertised job position',
        },  # Seniority Level
        columns[15]: {
            'type': 'str',
            'description': 'URL of company advertising job position',
        },  # Company URL
        columns[16]: {
            'type': 'str',
            'description': 'URL of advertised job position',
        },  # Job URL
        columns[17]: {
            'type': 'str',
            'description': 'IN DUTCH - time passed between job ad being posted and job ad being collected',
            'example': '2u',
        },  # Job Age
        columns[18]: {
            'type': 'str',
            'description': 'IN DUTCH - time passed between job ad being posted and job ad being collected',
            'example': '2u',
        },  # Job Age Number
        columns[19]: {
            'type': 'str',
            'description': 'date on which job ad was posted',
            'example': '2020-12-30',
        },  # Job Date
        columns[20]: {
            'type': 'str',
            'description': 'date on which job ad was collected',
            'example': '2020-12-30',
        },
    }

    if args['print_enabled'] is True:
        print(f'Saving Metadata dict to {code_dir}/metadata.json')
    with open(f'{code_dir}/metadata.json', 'w', encoding='utf8') as f:
        json.dump(metadata_dict, f)

    return metadata_dict


# %%
# Function to merge metadata
def save_metadata(df_jobs, df_file_name, save_path, args=get_args()):

    if (not df_jobs.empty) and (len(df_jobs != 0)):
        if args['print_enabled'] ==True:
            print('Attaching Metadata to existing DF.')
        metadata_dict = create_metadata()
        metadata_key = 'metadat.iot'
        metadata_dict_json = json.dumps(metadata_dict)
        df_jobs_pyarrow = pa.Table.from_pandas(df_jobs)
        existing_metadata = df_jobs_pyarrow.schema.metadata
        combined_metadata = {
            metadata_key.encode(): metadata_dict_json.encode(),
            **existing_metadata,
        }
        df_jobs_pyarrow = df_jobs_pyarrow.replace_schema_metadata(combined_metadata)
        if args['print_enabled'] ==True:
            print('Saving DF as .parquet file.')
        pq.write_table(
            df_jobs_pyarrow,
            save_path + df_file_name.replace('.csv', '_pyarrow.parquet'),
            compression='GZIP',
        )
    elif (df_jobs.empty) and (len(df_jobs == 0)):
        df_jobs_pyarrow = pa.Table.from_pandas(df_jobs)

    return df_jobs_pyarrow


# %%
# Clean df and drop duplicates and -1 for job description
def clean_df(
    df_jobs: pd.DataFrame,
    id_dict_new = False,
    int_variable: str = 'Job ID',
    str_variable: str = 'Job Description',
    gender: str = 'Gender',
    age: str = 'Age',
    language: str = 'en',
    reset=True,
    args=get_args(),
) -> pd.DataFrame:

    df_jobs.columns = df_jobs.columns.to_series().apply(lambda x: x.strip())
    df_jobs.dropna(axis=0, how='all', inplace=True)
    df_jobs.dropna(axis=1, how='all', inplace=True)
    df_jobs.drop(
        df_jobs.columns[
            df_jobs.columns.str.contains(
                'unnamed|index|level', regex=True, case=False, flags=re.I
            )
        ],
        axis=1,
        inplace=True,
    )
    df_jobs[int_variable] = df_jobs[int_variable].apply(str)

    if reset is True:
        df_jobs = set_gender_age_sects_lang(df_jobs, str_variable=str_variable, id_dict_new=id_dict_new)

    subset_list=[int_variable, str_variable, gender, age]
    print('Cleaning DF')
    df_jobs.drop_duplicates(
        subset=subset_list,
        keep='first',
        inplace=True,
        ignore_index=True,
    )

    df_jobs = df_jobs.loc[
        (
            df_jobs[str_variable]
            .swifter.progress_bar(args['print_enabled'])
            .progress_bar(args['print_enabled'])
            .apply(lambda x: isinstance(x, str))
        )
        & (df_jobs[str_variable] != -1)
        & (df_jobs[str_variable] != '-1')
        & (df_jobs[str_variable] != None)
        & (df_jobs[str_variable] != 'None')
        & (df_jobs[str_variable] != np.nan)
        & (df_jobs[str_variable] != 'nan')
    ]

    print('Detecting Language.')
    df_jobs = detect_language(df_jobs, str_variable)
    if 'Language' in df_jobs.columns:
        df_jobs = df_jobs.loc[(df_jobs['Language'] == str(language))]

    if 'Search Keyword' in df_jobs.columns:
        for w_keyword, r_keyword in keyword_trans_dict.items():
            df_jobs.loc[(df_jobs['Search Keyword'] == str(w_keyword)), 'Search Keyword'] = r_keyword

    df_jobs.reset_index(inplace=True, drop=True)

    return df_jobs


# %%
# Lang detect
def detect_language(df_jobs: pd.DataFrame, str_variable = 'Job Description', args=get_args()) -> pd.DataFrame:
    if args['print_enabled'] is True:
        print('Starting language detection...')

    # df_jobs['Language'] = language
    try:
        df_jobs['Language'] = df_jobs[str_variable].swifter.progress_bar(args['print_enabled']).apply(detect_language_helper)

    except Exception as e:
        if args['print_enabled'] is True:
            print('Language not detected.')
    else:
        if args['print_enabled'] is True:
            print('Language detection complete.')

    return df_jobs


# %%
def detect_language_helper(x, language='en'):

    x = ''.join([i for i in x if i not in list(string.punctuation)])

    if not x or x in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan'] or x.isspace() or x.replace(' ', '').isdigit():
        return 'NO LANGUAGE DETECTED'
    # try:
    #     lang = WhatTheLang().predict_lang(x)
    #     return lang if lang not in ['CANT_PREDICT', 'nl', language] else detect(x)
    # except ValueError:
    try:
        return detect(x)
    except langdetect.LangDetectException:
        return 'NO LANGUAGE DETECTED'


# %%
# Function to order categories
def categorize_df_gender_age(
    df,
):
    # Arrange Categories
    try:
        df['Gender'] = (
            df['Gender'].astype('category').cat.reorder_categories(order_gender, ordered=True)
        )

        df['Gender'] = pd.Categorical(
            df['Gender'], categories=order_gender, ordered=True
        )
    except ValueError as e:
        print(e)
    try:
        df['Age'] = df['Age'].astype('category').cat.reorder_categories(order_age, ordered=True)

        df['Age'] = pd.Categorical(df['Age'], categories=order_age, ordered=True)
    except ValueError as e:
        print(e)

    return df


# %%
# Function to dummy code gender and age
def dummy_code_df_gender_age(df, print_info=False, args=get_args()):
    # Gender Recode
    df.loc[df['Gender'] == 'Female', ['Gender_Female']] = 1
    df.loc[df['Gender'] != 'Female', ['Gender_Female']] = 0

    df.loc[df['Gender'] == 'Mixed Gender', ['Gender_Mixed']] = 1
    df.loc[df['Gender'] != 'Mixed Gender', ['Gender_Mixed']] = 0

    df.loc[df['Gender'] == 'Male', ['Gender_Male']] = 1
    df.loc[df['Gender'] != 'Male', ['Gender_Male']] = 0

    # Age Recode
    df.loc[df['Age'] == 'Older Worker', ['Age_Older']] = 1
    df.loc[df['Age'] != 'Older Worker', ['Age_Older']] = 0

    df.loc[df['Age'] == 'Mixed Age', ['Age_Mixed']] = 1
    df.loc[df['Age'] != 'Mixed Age', ['Age_Mixed']] = 0

    df.loc[df['Age'] == 'Younger Worker', ['Age_Younger']] = 1
    df.loc[df['Age'] != 'Younger Worker', ['Age_Younger']] = 0

    # Gender Recode
    df.loc[df['Gender'] == 'Female', ['Gender_Num']] = 1
    df.loc[df['Gender'] == 'Mixed Gender', ['Gender_Num']] = 2
    df.loc[df['Gender'] == 'Male', ['Gender_Num']] = 3

    # Age Recode
    df.loc[df['Age'] == 'Older Worker', ['Age_Num']] = 1
    df.loc[df['Age'] == 'Mixed Age', ['Age_Num']] = 2
    df.loc[df['Age'] == 'Younger Worker', ['Age_Num']] = 3

    if print_info is True:
        df_gender_age_info(df)

    return df


# %%
# Funtion to print df gender and age info
def df_gender_age_info(
    df,
    ivs_all=ivs_all,
):
    # Print Info
    print('\nDF INFO:\n')
    df.info()

    for iv in ivs_all:
        try:
            print('='*20)
            print(f'{iv}:')
            print('-'*20)
            print(f'{iv} Counts:\n{df[f"{iv}"].value_counts()}')
            print('-'*20)
            print(f'{iv} Percentages:\n{df[f"{iv}"].value_counts(normalize=True).mul(100).round(1).astype(float)}')
            try:
                print('-'*20)
                print(f'{iv} Mean: {df[f"{iv}"].mean().round(2).astype(float)}')
                print('-'*20)
                print(f'{iv} Standard Deviation: {df[f"{iv}"].std().round(2).astype(float)}')
            except Exception:
                pass
        except Exception:
            print(f'{iv} not available.')

    print('\n')


# %%
# Funtion to print df gender and age info
def df_warm_comp_info(
    df, dvs_all=['Warmth', 'Warmth_Probability', 'Competence', 'Competence_Probability'], print_info=False,
):
    # Print Info
    print('\nDF INFO:\n')
    df.info()

    if print_info is True:
        for dv in dvs_all:
            if '_Probability' not in dv:
                try:
                    print('='*20)
                    print(f'{dv}:')
                    print('-'*20)
                    print(f'{dv} Counts:\n{df[f"{dv}"].value_counts()}')
                    print('-'*20)
                    print(f'{dv} Percentages:\n{df[f"{dv}"].value_counts(normalize=True).mul(100).round(1).astype(float)}')
                    print('-'*20)
                    print(f'{dv} Means: {df[f"{dv}"].mean().round(2).astype(float)}')
                    print('-'*20)
                    print(f'{dv} Standard Deviation: {df[f"{dv}"].std().round(2).astype(float)}')
                except Exception:
                    print(f'{dv} not available.')

    print('\n')


# %%
# Function to plot df values
def value_count_df_plot(which_df, main_cols, num_unique_values=100, filter_lt_pct=.05, height=1300, width=1500, align='left'):
    cols = []
    nunique = []
    df = which_df #specify which df you're using
    which_df = df

    for c in which_df.columns:
        cols.append(c)
        nunique.append(which_df[c].nunique())
    df_cols = pd.DataFrame(nunique)
    df_cols['cols'] = cols
    df_cols.columns = ['nunique','column']
    df_cols = df_cols[['column','nunique']]
    df_cols_non_unique = df_cols[
            (df_cols['nunique'] <= df_cols.shape[0])
            & (df_cols['nunique'] > 1)
        ].sort_values(by='nunique',ascending=True)
    merch_cols = list(df_cols_non_unique.column)
    # print(df_cols_non_unique.shape)
    df_cols_non_unique.head()

    #lte 30 unique values in any column
    num_unique_values = num_unique_values
    df_cols_non_unique = df_cols_non_unique[df_cols_non_unique['nunique'] <= num_unique_values]
    list_non_unique_cols = list(df_cols_non_unique['column'])
    print('total number of cols with lte', num_unique_values, 'unique values:', len(list_non_unique_cols))

    #include main cols
    main_cols = [main_cols]
    number_of_main_cols = len(main_cols)

    #append main cols with interesting cols
    interestin_cols = main_cols + list_non_unique_cols

    #specify interesting cols
    df1 = df.loc[:, df.columns.isin(interestin_cols)]
    df1 = df1.iloc[:,:-1]

    #get value counts for each value in each col
    def value_counts_col(df,col):
        df = df
        value_counts_df = pd.DataFrame(round(df[col].value_counts(normalize=True),2).reset_index())
        value_counts_df.columns = ['value','value_counts']
        value_counts_df['feature'] = col
        return value_counts_df

    all_cols_df = []
    for i in df1.columns[number_of_main_cols:]:
        dfs = value_counts_col(df1,i)
        all_cols_df.append(dfs)

    #append column values to end of column
    which_df = pd.concat(all_cols_df)
    which_df['value'] = which_df['value'].fillna('null')
    which_df['feature_value'] = which_df['feature'] + "_" + which_df['value'].map(str)
    which_df = which_df.drop(['value','feature'],axis=1)
    which_df = which_df[['feature_value','value_counts']]
    which_df = which_df.sort_values(by='value_counts',ascending=False)

    #filter out less than x% features
    filter_lt_pct = filter_lt_pct
    which_df = which_df[which_df['value_counts'] >= filter_lt_pct]

    print('df shape:', which_df.shape,'\n')

    #table plot
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=list(which_df.columns)
                    ,fill_color='black'
                    ,align=align
                    ,font=dict(color='white', size=12)
                    ,line_color=['darkgray','darkgray','darkgray']
                   )
        ,cells=dict(values=[which_df['feature_value'],which_df['value_counts']]
                   ,fill_color='black'
                   ,align=align
                   ,font=dict(color='white', size=12)
                   ,line_color=['darkgray','darkgray','darkgray']
                   )
    )
    ])
    fig2.update_layout(height=100, margin=dict(r=0, l=0, t=0, b=0))
    fig2.show()

    #bar plot
    fig1 = px.bar(which_df
                     ,y='feature_value'
                     ,x='value_counts'
                     ,height=height
                     ,width=width
                     ,template='plotly_dark'
                     ,text='value_counts'
                     ,title='<b>Outlier Values of Interesting Cols within Dataset'
                    )
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(showgrid=False)
    fig1.show()

    # return fig1,fig2


# %% Function to visaulize
def get_viz(df_name, df_df, dataframes, args=get_args()):
    from setup_module.params import analysis_columns, image_save_format

    # Visualize data balance
    dataframes[df_name]['Warmth'].value_counts()
    dataframes[df_name]['Competence'].value_counts()
    warm_comp_count = (
        dataframes[df_name][analysis_columns]
        .reset_index()
        .groupby(analysis_columns)
        .count()
        .sort_values(by='index')
    )
    fig, ax = plt.subplots()
    fig.suptitle(f'{df_name}: Warmth and Competence Sentence Counts', fontsize=16.0)
    warm_comp_count.plot(kind='barh', stacked=True, legend=True, color='blue', ax=ax).grid(
        axis='y'
    )
    if args['save_enabled'] is True:
        fig.savefig(f'{args["plot_save_path"]}{df_name} - Warmth and Competence Sentence Counts.{image_save_format}', format=image_save_format, dpi=3000)

    fig.show()
    plt.pause(0.1)


# %%
def set_language_requirement(
    df_jobs,
    str_variable = 'Job Description',
    dutch_requirement_pattern = r'[Ll]anguage: [Dd]utch|[Dd]utch [Pp]referred|[Dd]utch [Re]quired|[Dd]utch [Ll]anguage|[Pp]roficient in [Dd]utch|[Ss]peak [Dd]utch|[Kk]now [Dd]utch',
    english_requirement_pattern = r'[Ll]anguage: [Ee]nglish|[Ee]nglish [Pp]referred|[Ee]nglish [Re]quired|[Ee]nglish [Ll]anguage|[Pp]roficient in [Ee]nglish|[Ss]peak [Ee]nglish|[Kk]now [Ee]nglish',
    args=get_args(),
    ):
    # Language requirements
    # Dutch
    print('Setting Dutch language requirements.')
    if 'Dutch Requirement' in df_jobs.columns:
        df_jobs.drop(columns=['Dutch Requirement'], inplace=True)
    df_jobs['Dutch Requirement'] = np.where(
        df_jobs[str_variable].str.contains(dutch_requirement_pattern),
        'Yes',
        'No',
    )

    # English
    print('Setting English language requirements.')
    if 'English Requirement' in df_jobs.columns:
        df_jobs.drop(columns=['English Requirement'], inplace=True)
    df_jobs['English Requirement'] = np.where(
        df_jobs[str_variable].str.contains(english_requirement_pattern),
        'Yes',
        'No',
    )

    return df_jobs


# %%
def set_sector_and_percentage(
    df_jobs,
    sector_dict_new=False,
    sectors_path = f'{code_dir}/data/content analysis + ids + sectors/Sectors + Age and Gender Composition of Industires and Jobs/Output CSVs for Occupational Sectors/ALL SECTOR AGE AND GENDER TABLE.xlsx',
    age_limit = 45,
    age_ratio = 10,
    gender_ratio = 20,
    args=get_args(),
):

    sbi_english_keyword_list = args['sbi_english_keyword_list']
    sbi_english_keyword_dict = args['sbi_english_keyword_dict']
    sbi_sectors_dict = args['sbi_sectors_dict']
    sbi_sectors_dict_full = args['sbi_sectors_dict_full']
    sbi_sectors_dom_gen = args['sbi_sectors_dom_gen']
    sbi_sectors_dom_age = args['sbi_sectors_dom_age']
    trans_keyword_list = args['trans_keyword_list']
    df_sectors = get_sector_df_from_cbs()

    if sector_dict_new is True:
        sector_vs_job_id_dict = make_job_id_v_sector_key_dict()
    elif sector_dict_new is False:
        with open(validate_path(f'{args["parent_dir"]}job_id_vs_sector_all.json'), encoding='utf-8') as f:
            sector_vs_job_id_dict = json.load(f)

    # Set Sectors
    print('Setting sector.')
    if 'Sector' in df_jobs.columns:
        df_jobs.drop(columns=['Sector'], inplace=True)
    for sect, sect_dict in sector_vs_job_id_dict.items():
        for keyword, job_ids in sect_dict.items():
            df_jobs.loc[df_jobs['Job ID'].astype(str).apply(lambda x: x.lower().strip()).isin([str(i) for i in job_ids]), 'Sector'] = str(sect).lower().strip()
    # if 'Search Keyword' in df_jobs.columns:
    #     if df_jobs['Sector'].isnull().values.any() or df_jobs['Sector'].isnull().sum() > 0 or df_jobs['Sector'].isna().values.any() or df_jobs['Sector'].isna().sum() > 0:
    #         df_sectors = get_sector_df_from_cbs()
    #         for idx, row in df_sectors.iterrows():
    #             if isinstance(row[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')], list) and df_jobs['Job ID'].astype(str).apply(lambda x: x.lower().strip()).isin([str(i) for i in row[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Keywords')]]):
    #                     print(row[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name')])

    print('Setting sector code and percentages.')
    # Add gender and age columns
    sect_cols = ['Sector Code', '% Female', '% Male', '% Older', '% Younger']
    for col in sect_cols:
        if col in df_jobs.columns:
            df_jobs.drop(columns=col, inplace=True)
    df_jobs = df_jobs.reindex(columns=[*df_jobs.columns, *sect_cols], fill_value=np.nan)

    # Set Percentages
    # Open df
    df_sectors = get_sector_df_from_cbs()
    for index, row in df_jobs.iterrows():
        for idx, r in df_sectors.iterrows():
            if str(row['Sector']).strip().lower() == str(r[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Sector Name')]).strip().lower():
                df_jobs.loc[index, 'Sector Code'] = r[('SBI Sector Titles', 'Industry class / branch (SIC2008)', 'Code')]
                df_jobs.loc[index, '% Female'] = r[('Gender', 'Female', '% per Sector')]
                df_jobs.loc[index, '% Male'] = r[('Gender', 'Male', '% per Sector')]
                df_jobs.loc[index, '% Older'] = r[('Age', f'Older (>= {age_limit} years)', '% per Sector')]
                df_jobs.loc[index, '% Younger'] = r[('Age', f'Younger (< {age_limit} years)', '% per Sector')]

    print('Done setting sector percentages.')

    return df_jobs


# # %%
# def set_sector_and_percentage_helper(df_jobs, keyword, trans_keyword_list, args=get_args()):

#     for index, row in df_jobs.iterrows():
#         if row['Sector'] in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
#             df_jobs.loc[(df_jobs['Job ID'].astype(str).apply(lambda x: x.strip().lower()).isin([i.strip().lower() for i in job_ids if isinstance (i, str)])) | (df_jobs['Search Keyword'].astype(str).apply(lambda x: x.strip().lower()) == keyword.strip().lower()), 'Sector'] = sect.capitalize()
#             for index, row in df_jobs.iterrows():
#                 if row['Sector'] in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
#                     df_jobs.loc[(df_jobs['Search Keyword'].astype(str).apply(lambda x: x.strip().lower()) == keyword.strip().lower()), 'Sector'] = sect.capitalize()
#                     if row['Sector'] in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
#                         trans_keyword_list.append(keyword.strip().lower())
#                         trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

#     return df_jobs


# %%
def set_gender_age(
    df_jobs,
    id_dict_new=False,
    args=get_args(),
):

    sbi_english_keyword_list = args['sbi_english_keyword_list']
    sbi_english_keyword_dict = args['sbi_english_keyword_dict']
    sbi_sectors_dict = args['sbi_sectors_dict']
    sbi_sectors_dict_full = args['sbi_sectors_dict_full']
    sbi_sectors_dom_gen = args['sbi_sectors_dom_gen']
    sbi_sectors_dom_age = args['sbi_sectors_dom_age']
    trans_keyword_list = args['trans_keyword_list']

    if id_dict_new is True:
        job_id_dict = make_job_id_v_genage_key_dict()

    elif id_dict_new is False:
        with open(validate_path(f'{args["parent_dir"]}job_id_vs_all.json'), encoding='utf8') as f:
            job_id_dict = json.load(f)

    print('Setting gender and age.')
    # Add gender and age columns
    gen_age_cols = ['Gender', 'Age']
    for col in gen_age_cols:
        if col in df_jobs.columns:
            df_jobs.drop(columns=col, inplace=True)
    df_jobs = df_jobs.reindex(columns=[*df_jobs.columns, *gen_age_cols], fill_value=np.nan)

    # Gender
    print('Setting gender.')
    try:
        for sect, cat in sbi_sectors_dom_gen.items():
            df_jobs.loc[df_jobs['Sector'].astype(str).apply(lambda x: x.lower().strip()) == str(sect).lower().strip(), 'Gender'] = str(cat)
    except Exception as e:
        for cat in ['Mixed Gender', 'Male', 'Female']:
            df_jobs.loc[df_jobs['Job ID'].astype(str).apply(lambda x: x.lower().strip()).isin([str(i) for i in job_id_dict[cat]]), 'Gender'] = str(cat)

    # Age
    print('Setting age.')
    try:
        for sect, cat in sbi_sectors_dom_age.items():
            df_jobs.loc[df_jobs['Sector'].astype(str).apply(lambda x: x.lower().strip()) == str(sect).lower().strip(), 'Age'] = str(cat)
    except Exception as e:
        for cat in ['Mixed Age', 'Younger Worker', 'Older Worker']:
            df_jobs.loc[df_jobs['Job ID'].astype(str).apply(lambda x: x.lower().strip()).isin([str(i) for i in job_id_dict[cat]]), 'Age'] = str(cat)

    print('Categorizing gender and age')
    df_jobs = categorize_df_gender_age(df_jobs)
    df_jobs = dummy_code_df_gender_age(df_jobs)

    print('Done setting gender and age.')

    return df_jobs


# %%
def set_gender_age_sects_lang(df_jobs, str_variable, id_dict_new=False, args=get_args()):

    # now = datetime.datetime.now().total_seconds()

    df_jobs=set_language_requirement(df_jobs, str_variable=str_variable)
    # print(f'Time taken for set_language_requirement: {now-datetime.datetime.now().total_seconds()}')
    # now = datetime.datetime.now().total_seconds()
    df_jobs=set_sector_and_percentage(df_jobs, sector_dict_new=id_dict_new)
    # print(f'Time taken for set_sector_and_percentage: {now-datetime.datetime.now().total_seconds()}')
    # now = datetime.datetime.now().total_seconds()
    df_jobs=set_gender_age(df_jobs, id_dict_new=id_dict_new)
    # print(f'Time taken for set_gender_age: {now-datetime.datetime.now().total_seconds()}')

    return df_jobs


# %%
# Load and merge existing dict and df
def load_merge_dict_df(
    keyword: str,
    save_path: str,
    df_file_name: str,
    json_file_name: str,
    args=get_args()
):
    # df_jobs
    if is_non_zero_file(save_path + df_file_name.lower()) is True:
        if args['print_enabled'] is True:
            print(
                f'A DF with the name "{df_file_name.lower()}" already exists at {save_path}.\nNew data will be appended to the file.'
            )
        df_old_jobs = pd.read_csv(save_path + df_file_name.lower())
        if not df_old_jobs.empty:
            df_old_jobs = clean_df(df_old_jobs)
        else:
            print(f'{df_file_name} is empty!')

    elif is_non_zero_file(save_path + df_file_name.lower()) is False:
        if args['print_enabled'] is True:
            print(f'No DF with the name "{df_file_name.lower()}" found.')
        df_old_jobs = pd.DataFrame()
    if args['print_enabled'] is True:
        print(f'Old jobs DF of length: {df_old_jobs.shape[0]}.')

    # jobs
    if is_non_zero_file(save_path + json_file_name.lower()) is True:
        if args['print_enabled'] is True:
            print(
                f'A list of dicts with the name "{json_file_name.lower()}" already exists at {save_path}.\nNew data will be appended to this file.'
            )
        with open(save_path + json_file_name, encoding='utf8') as f:
            old_jobs = json.load(f)
        # old_jobs = remove_dupe_dicts(old_jobs)
    elif is_non_zero_file(save_path + json_file_name.lower()) is False:
        if args['print_enabled'] is True:
            print(f'No list of dicts with the name "{json_file_name.lower()}" found.')
        old_jobs = []
    if args['print_enabled'] is True:
        print(f'Old jobs dict of length: {len(old_jobs)}.')

    # Merge dicts and df
    if df_old_jobs is not None:
        # Convert old df to jobs
        try:
            jobs_from_df_old_jobs = df_old_jobs.reset_index().to_dict('records')
        except Exception:
            jobs_from_df_old_jobs = df_old_jobs.reset_index(drop=True).to_dict(
                'records'
            )

        # Merge jobs from df to jobs from file
        if args['print_enabled'] is True:
            print('Merging DF with jobs into new list.')
        old_jobs.extend(jobs_from_df_old_jobs)
        # old_jobs = remove_dupe_dicts(old_jobs)
        jobs = []
        for myDict in old_jobs:
            if myDict not in jobs:
                jobs.append(myDict)

        if is_non_zero_file(save_path + df_file_name.lower()) is True or is_non_zero_file(save_path + json_file_name.lower()) is True:
            with open(save_path + json_file_name, 'w', encoding='utf8') as f:
                json.dump(jobs, f)
    elif df_old_jobs is None:
        jobs = old_jobs

    if args['print_enabled'] is True:
        print('-' * 20)
        if len(jobs) > 0:
            print(
                f'List of dicts of length {len(jobs)} was loaded for {jobs[0]["Search Keyword"]}.'
            )

        elif len(jobs) == 0:
            print(f'List of dicts of length {len(jobs)} was loaded for {keyword}.')
        print('-' * 20)

    return jobs, df_old_jobs


# %%
# Function to save df as csv
def save_df(
    keyword: str,
    df_jobs,
    save_path: str,
    keyword_file: str,
    df_file_name: str,
    print_enabled: bool = False,
    clean_enabled: bool = True,
    args=get_args(),
):
    if print_enabled is True:
        print(f'Saving {keyword} jobs data to df...')

    if (not df_jobs.empty) and (len(df_jobs != 0)):
        if print_enabled is True:
            print(f'Cleaning {keyword} df.')

        # Drop duplicates and -1 for job description
        if clean_enabled is True:
            df_jobs = clean_df(df_jobs)

        # Search keyword
        try:
            search_keyword = df_jobs['Search Keyword'].iloc[0].lower().replace("-Noon's MacBook Pro",'')
        except KeyError:
            df_jobs.reset_index(drop=True, inplace=True)
            search_keyword = df_jobs['Search Keyword'].iloc[0].lower().replace("-Noon's MacBo an and.  ok Pro",'')
        except IndexError:
            print(len(df_jobs))

        # Save df to csv
        if print_enabled is True:
            print(f'Saving {keyword.lower()} jobs df of length {len(df_jobs.index)} to csv as {df_file_name.lower()} in location {save_path}')

        df_jobs.to_csv(save_path + df_file_name, mode='w', sep=',', header=True, index=True)
        df_jobs.to_csv(save_path + df_file_name.split(args["file_save_format_backup"])[0]+'txt', mode='w', sep=',', header=True, index=True)

        if (not df_jobs.empty) and (len(df_jobs != 0)):
            try:
                df_jobs_pyarrow = save_metadata(df_jobs, df_file_name, save_path)
            except Exception:
                pass

    elif df_jobs.empty:
        if print_enabled is True:
            print(f'Jobs DataFrame is empty since no jobs results were found for {str(keyword)}. Moving on to next search.')

    return df_jobs


# %%
# Post collection cleanup
def site_loop(site, site_list, site_from_list, args, df_list_from_site=None):

    if site_from_list is True:
        df_list_from_site = []
        for site in tqdm.tqdm(site_list):
            if args['print_enabled'] is True:
                print('-' * 20)
                print(f'Cleaning up LIST OF DFs for {site}.')
            glob_paths = glob.glob(f'{scraped_data}/{site}/Data/*.json')+glob.glob(f'{scraped_data}/{site}/Data/*.csv')+glob.glob(f'{scraped_data}/{site}/Data/*.xlsx')

            yield site, df_list_from_site, glob_paths

    elif site_from_list is False:
        df_list_from_site = None
        if args['print_enabled'] is True:
            print('-' * 20)
            print('Cleaning up LIST OF DFs from all sites.')
        glob_paths = glob.glob(f'{scraped_data}/*/Data/*.json')+glob.glob(f'{scraped_data}/*/Data/*.csv')+glob.glob(f'{scraped_data}/*/Data/*.xlsx')

        yield site, df_list_from_site, glob_paths


# %%
def site_save(site, df_jobs, args, chunk_size = 1024 * 1024):
    if args['save_enabled'] is True:
        print(f'Saving df_{site}_all_jobs.{args["file_save_format"]}')
        with open(args["df_dir"] + f'df_{site}_all_jobs.{args["file_save_format"]}', 'wb') as f:
            pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done saving df_{site}_all_jobs.{args["file_save_format"]}')


# %%
def keyword_loop(keyword, keywords_from_list, glob_paths, args, translator, df_list_from_keyword=None):

    if keywords_from_list is True:
        df_list_from_keyword = []
        for glob_path in glob_paths:
            if 'dict_' in glob_path and glob_path.endswith('.json'):
                keyword = glob_path.split('dict_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
            elif 'df_' in glob_path and (glob_path.endswith('.csv') or glob_path.endswith('.xlsx')):
                keyword = glob_path.split('df_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
            if '_' in keyword:
                keyword = ' '.join(keyword.split('_')).strip().lower()

            if args['print_enabled'] is True:
                print(f'Post collection cleanup for {keyword}.')
            yield keyword, df_list_from_keyword

    elif keywords_from_list is False:
        keyword=keyword
        df_list_from_keyword = None
        if args['print_enabled'] is True:
            print(f'Post collection cleanup for {keyword}.')
        yield keyword, df_list_from_keyword


# %%
def keyword_save(keyword, site, df_jobs, args):
    if args['save_enabled'] is True:
        print(f'Saving df_{site}_{keyword}_all_jobs.{args["file_save_format"]}')
        with open(args['df_dir'] + f'df_{site}_{keyword}_all_jobs.{args["file_save_format"]}', 'wb') as f:
            pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done saving df_{site}_{keyword}_all_jobs.{args["file_save_format"]}')


# %%
def post_cleanup(
    site_from_list=True,
    keywords_from_list=True,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    job_id_save_enabled=False,
    job_sector_save_enabled=False,
    keyword='',
    site='',
    keywords_list=None,
    all_save_path = f'job_id_vs_all.json',
    args=get_args(),
    translator = Translator(),
    translate_keywords=True,
):

    print(
        f'NOTE: The function "post_cleanup" contains the following optional (default) arguments:\n{get_default_args(post_cleanup)}'
    )
    print('-' * 20)

    if keywords_list is None:
        keywords_list = []

    # Get original collected sectors
    trans_keyword_list = get_trans_keyword_list()

    for site, df_list_from_site, glob_paths in tqdm.tqdm(site_loop(site=site, site_list=site_list, site_from_list=site_from_list, args=args)):
        for keyword, df_list_from_keyword in tqdm.tqdm(keyword_loop(keyword=keyword, keywords_from_list=keywords_from_list, glob_paths=glob_paths, args=args, translator=translator)):

            trans_keyword = keyword.strip().lower()

            if translate_keywords is True and detect(trans_keyword) != 'en':
                while True:
                    try:
                        trans_keyword = translator.translate(trans_keyword).text.strip().lower()
                    except Exception as e:
                        time.sleep(0.3)
                        continue
                    break

            for w_keyword, r_keyword in keyword_trans_dict.items():
                if str(trans_keyword.lower()) == w_keyword.lower():
                    trans_keyword = r_keyword.strip().lower()
                trans_keyword_list.append(trans_keyword)

            try:
                df_jobs = post_cleanup_helper(keyword, site)
                print(f'DF {trans_keyword.title()} collected.')

                if df_jobs.empty and args['print_enabled'] is True:
                    print(f'DF {trans_keyword.title()} not collected yet.')

            except Exception:
                if args['print_enabled'] is True:
                    print(f'An error occured with finding DF {keyword}.')
                df_jobs = pd.DataFrame()

            else:
                if args['print_enabled'] is True:
                    print(f'Cleaning up LIST OF DFs for {keyword}.')

                if site_from_list is True and keywords_from_list is True:
                    if (not df_jobs.empty) and (len(df_jobs != 0)):
                        df_list_from_keyword.append(df_jobs)
                    df_list_from_site.append(df_list_from_keyword)
                    df_jobs = df_list_from_site
                    # site_save(site, df_jobs, args=args)
                elif site_from_list is True and keywords_from_list is False:
                    if (not df_jobs.empty) and (len(df_jobs != 0)):
                        df_list_from_site.append(df_jobs)
                    df_jobs = df_list_from_site
                    # site_save(site, df_jobs, args=args)
                elif site_from_list is False and keywords_from_list is True:
                    if (not df_jobs.empty) and (len(df_jobs != 0)):
                        df_list_from_keyword.append(df_jobs)
                    df_jobs = df_list_from_keyword
                #     keyword_save(keyword, site, df_jobs, args=args)
                # elif site_from_list is False and keywords_from_list is False:
                #     keyword_save(keyword, site, df_jobs, args=args)

    for lst in df_jobs:
        for df in lst:
            if isinstance(df, DataFrame):
                df = set_gender_age_sects_lang(df)

    if translate_keywords is True:
        trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

    print(f'Saving df_jobs_post_cleanup.{args["file_save_format"]}')
    with open(args["df_dir"] + f'df_jobs_post_cleanup.{args["file_save_format"]}', 'wb') as f:
        pickle.dump(df_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done saving df_jobs_post_cleanup.{args["file_save_format"]}')

    # print(f'Saving df_jobs_post_cleanup.{args["file_save_format_backup"]}')
    # with open(df_dir + f'df_jobs_post_cleanup.{file_save_format_backup}', 'w', newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(df_jobs)
    # print(f'Done saving df_jobs_post_cleanup.{args["file_save_format_backup"]}')

    if job_id_save_enabled is True:
        job_id_dict = make_job_id_v_genage_key_dict(site_from_list=False)
    if job_sector_save_enabled is True:
        sector_vs_job_id_dict = make_job_id_v_sector_key_dict(site_from_list=False)

    return df_jobs


# %%
# Get keywords and files for post collection df cleanup
def post_cleanup_helper(keyword, site, args=get_args()):
    (
        keyword_url,
        keyword_file,
        save_path,
        json_file_name,
        df_file_name,
        logs_file_name,
        filemode,
    ) = main_info(keyword, site)

    jobs, df_old_jobs = load_merge_dict_df(
        keyword, save_path, df_file_name, json_file_name
    )

    if is_non_zero_file(save_path + df_file_name.lower()) is True or is_non_zero_file(save_path + json_file_name.lower()) is True:
        with open(save_path + json_file_name, 'w', encoding='utf8') as f:
            json.dump(jobs, f)
        df_jobs = pd.DataFrame(jobs)
        if (not df_jobs.empty) and (len(df_jobs != 0)):
            # Save df as csv
            if args['save_enabled'] is True:
                df_jobs = save_df(
                    keyword,
                    df_jobs,
                    save_path,
                    keyword_file.lower(),
                    df_file_name.lower(),
                )
        elif (df_jobs.empty) or (len(df_jobs == 0)):
            if args['print_enabled'] is True:
                print(
                    f'Jobs DataFrame is empty since no jobs results were found for {str(keyword)}.'
                )

    elif is_non_zero_file(save_path + df_file_name.lower()) is False or is_non_zero_file(save_path + json_file_name.lower()) is False:
        if args['print_enabled'] is True:
            print(f'No jobs file found for {keyword} in path: {save_path}.')
        df_jobs = pd.DataFrame()

    return df_jobs


# %%
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# %%
# Function to clean from old folder
def clean_from_old(
    site=None,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    files = None,
    exten_to_find = ['.json','.csv','.xlsx'],
    translator = Translator(),
    args=get_args(),
):
    if files is None:
        files = []

    if site is None and files == []:
        try:
            for site in site_list:
                for file_ in glob.glob(f'{scraped_data}/{site}/Data/*.json')+glob.glob(f'{scraped_data}/{site}/Data/*.csv')+glob.glob(f'{scraped_data}/{site}/Data/*.xlsx'):
                    files.append(file)
        except Exception as e:
            for file_ in glob.glob(f'{scraped_data}/*/Data/*.json')+glob.glob(f'{scraped_data}/*/Data/*.csv')+glob.glob(f'{scraped_data}/*/Data/*.xlsx'):
                files.append(file_)

    elif site is not None and files == []:
        for file_ in glob.glob(f'{scraped_data}/{site}/Data/*.json')+glob.glob(f'{scraped_data}/{site}/Data/*.csv')+glob.glob(f'{scraped_data}/{site}/Data/*.xlsx'):
            files.append(file_)

    for file_ in tqdm.tqdm(files):
        if site is None:
            site = file_.split(f'{code_dir}/')[1].split('/Data')[0].strip()
        if 'dict_' in file_ and file_.endswith('.json'):
            keyword = file_.split('dict_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
        elif 'df_' in file_ and (file_.endswith('.csv') or file_.endswith('.xlsx')):
            keyword = file_.split('df_')[1].split('.')[0].replace("-Noon's MacBook Pro",'').strip().lower()
        if '_' in keyword:
            keyword = ' '.join(keyword.split('_')).strip().lower()

        if detect(keyword) != 'en':
            while True:
                try:
                    trans_keyword = translator.translate(keyword).text.strip().lower()
                except Exception as e:
                    time.sleep(0.3)
                    continue
                break

        else:
            trans_keyword = keyword

        for w_keyword, r_keyword in keyword_trans_dict.items():
            if trans_keyword and trans_keyword != keyword:
                if str(trans_keyword.strip().lower()) == w_keyword.strip().lower():
                    trans_keyword = r_keyword.strip().lower()
                else:
                    trans_keyword = trans_keyword.strip().lower()

        print(f'Getting data for {trans_keyword}.')
        if trans_keyword != keyword:
            print(f'Translated from: {keyword}.')

        (
            keyword_url,
            keyword_file,
            save_path,
            json_file_name,
            df_file_name,
            logs_file_name,
            filemode,
        ) = main_info(keyword, site)
        if trans_keyword != keyword:
            (
                trans_keyword_url,
                trans_keyword_file,
                trans_save_path,
                trans_json_file_name,
                trans_df_file_name,
                trans_logs_file_name,
                trans_filemode,
            ) = main_info(trans_keyword, site)

        if is_non_zero_file(file_) is True:
            df_jobs = pd.DataFrame()

            if file_.endswith('.json'):
                try:
                    df_jobs_json = pd.read_json(file_, orient='records')
                except ValueError:
                    with open(file_) as f:
                        df_jobs_json = pd.DataFrame(json.load(f))
                df_jobs = df_jobs.append(df_jobs_json, ignore_index=True)
                if trans_keyword != keyword and is_non_zero_file(trans_save_path + trans_json_file_name.lower()) is True:
                    trans_df_jobs_json = pd.read_json(trans_save_path + trans_json_file_name.lower(), orient='records')
                    df_jobs = df_jobs.append(trans_df_jobs_json, ignore_index=True)

            if file_.endswith('.csv'):
                df_jobs_csv = pd.read_csv(file_)
                df_jobs = df_jobs.append(df_jobs_csv, ignore_index=True)
                if trans_keyword != keyword and is_non_zero_file(trans_save_path + trans_df_file_name.lower()) is True:
                    trans_df_jobs_csv = pd.read_csv(trans_save_path + trans_df_file_name.lower())
                    df_jobs = df_jobs.append(trans_df_jobs_csv, ignore_index=True)

            if file_.endswith('.xlsx'):
                df_jobs_xlsx = pd.read_excel(file_)
                df_jobs = df_jobs.append(df_jobs_xlsx, ignore_index=True)
                if trans_keyword != keyword and is_non_zero_file(trans_save_path + trans_df_file_name.lower()) is True:
                    trans_df_jobs_xlsx = pd.read_excel(trans_save_path + trans_df_file_name.lower().replace('csv', 'xlsx'))
                    df_jobs = df_jobs.append(trans_df_jobs_xlsx, ignore_index=True)

            if (not df_jobs.empty) and (len(df_jobs != 0)):
                df_jobs = clean_df(df_jobs)
                jobs = df_jobs.to_dict(orient='records')

                if is_non_zero_file(save_path + df_file_name.lower()) is True or is_non_zero_file(save_path + json_file_name.lower()) is True:
                    with open(save_path + json_file_name, 'w', encoding='utf8') as f:
                        json.dump(jobs, f)

                df_jobs = save_df(
                    keyword=keyword,
                    df_jobs=df_jobs,
                    save_path=save_path,
                    keyword_file=keyword_file.lower(),
                    df_file_name=df_file_name.lower(),
                    clean_enabled = False,
                )

        else:
            print(f'Data for {site} {keyword} is empty.')

    return df_jobs


# %%
# Function to match job sector to larger sectors
def make_job_id_v_sector_key_dict(
    site_from_list=False,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    all_save_path = f'job_id_vs_sector',
    args=get_args(),
    ):

    print(
        f'NOTE: The function "make_job_id_v_sector_key_dict" contains the following optional (default) arguments:\n{get_default_args(make_job_id_v_sector_key_dict)}'
    )

    sib_5_loc = validate_path(f'{args["parent_dir"]}Sectors + Age and Gender Composition of Industires and Jobs/Found Data for Specific Occupations/SBI_ALL_NACE_REV2.csv')

    # Get keywords and paths to df_jobs
    if site_from_list is True:
        for site in site_list:
            if args['print_enabled'] is True:
                print(f'Getting job ids for {site}.')
            df_jobs_paths = list((glob.glob(f'{scraped_data}/{site}/Data/*.csv')))
            sector_vs_job_id_dict = make_job_id_v_sector_key_dict_helper(
                df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_{site}',
            )
    elif site_from_list is False:
        df_jobs_paths = glob.glob(f'{scraped_data}/*/Data/*.csv')
        sector_vs_job_id_dict = make_job_id_v_sector_key_dict_helper(
            df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_all',
        )

    return sector_vs_job_id_dict


# %%
def make_job_id_v_sector_key_dict_helper(
    df_jobs_paths,
    all_save_path,
    sector_vs_job_id_dict=defaultdict(lambda: defaultdict(list)),
    args=get_args(),
):

    sbi_english_keyword_list = args['sbi_english_keyword_list']
    sbi_english_keyword_dict = args['sbi_english_keyword_dict']
    sbi_sectors_dict = args['sbi_sectors_dict']
    sbi_sectors_dict_full = args['sbi_sectors_dict_full']
    sbi_sectors_dom_gen = args['sbi_sectors_dom_gen']
    sbi_sectors_dom_age = args['sbi_sectors_dom_age']
    trans_keyword_list = args['trans_keyword_list']

    for path in df_jobs_paths:
        df_jobs = pd.read_csv(path)

        for index, row in tqdm.tqdm(df_jobs.iterrows()):
            if row['Search Keyword'] not in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
                search_keyword = str(row['Search Keyword'].strip().lower().replace("-Noon's MacBook Pro",'').strip().lower())
                for w_keyword, r_keyword in keyword_trans_dict.items():
                    if search_keyword == w_keyword.lower():
                        df_jobs.loc[index, 'Search Keyword'] = r_keyword.strip().lower()
                        df_jobs.to_csv(path)
                trans_keyword_list.append(search_keyword)

                for sector, keywords_list in args['sbi_sectors_dict_full'].items():
                    if search_keyword in str(keywords_list):
                        sector_vs_job_id_dict[str(sector)][str(search_keyword)].append(str(row['Job ID']))

                # for code, sect_dict in sbi_sectors_dict.items():
                #     if str(row['Search Keyword']) in str(sect_dict['Used_Sector_Keywords']):
                #         sector_vs_job_id_dict[str(sect_dict['Sector_Name'])][str(row['Search Keyword'])].append(row['Job ID'])

    trans_keyword_list = save_trans_keyword_list(trans_keyword_list)

    if args['save_enabled'] is True:
        with open(f'{args["parent_dir"]}{all_save_path}.json', 'w', encoding='utf8') as f:
            json.dump(sector_vs_job_id_dict, f)
    elif args['save_enabled'] is False:
        print('No job id matching save enabled.')

    return sector_vs_job_id_dict


# %%
# Function to match job IDs with gender and age in dict
def make_job_id_v_genage_key_dict(
    site_from_list=False,
    site_list=['Indeed', 'Glassdoor', 'LinkedIn'],
    all_save_path = f'job_id_vs',
    args=get_args()):

    print(
        f'NOTE: The function "make_job_id_v_genage_key_dict" contains the following optional (default) arguments:\n{get_default_args(make_job_id_v_genage_key_dict)}'
    )

    # Get keywords and paths to df_jobs
    if site_from_list is True:
        for site in tqdm.tqdm(site_list):
            if args['print_enabled'] is True:
                print(f'Getting job ids for {site}.')
            df_jobs_paths = list((glob.glob(f'{scraped_data}/{site}/Data/*.csv')))
            job_id_dict = make_job_id_v_genage_key_dict_helper(
                df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_{site}'
            )
    elif site_from_list is False:
        df_jobs_paths = glob.glob(f'{scraped_data}/*/Data/*.csv')
        job_id_dict = make_job_id_v_genage_key_dict_helper(
            df_jobs_paths=df_jobs_paths, all_save_path=f'{all_save_path}_all'
        )

    return job_id_dict


# %%
def make_job_id_v_genage_key_dict_helper(
    df_jobs_paths,
    all_save_path,
    job_id_dict=defaultdict(list),
    args=get_args(),
):

    trans_keyword_list = args['trans_keyword_list']

    for path in tqdm.tqdm(df_jobs_paths):
        df_jobs = pd.read_csv(path)
        for index, row in tqdm.tqdm(df_jobs.iterrows()):

            if row['Search Keyword'] not in [None, 'None', '', ' ', [], -1, '-1', 0, '0', 'nan', np.nan, 'Nan']:
                search_keyword = str(row['Search Keyword'].replace("-Noon's MacBook Pro",'').strip().lower())
                for w_keyword, r_keyword in keyword_trans_dict.items():
                    if search_keyword == w_keyword.lower():
                        df_jobs.loc[index, 'Search Keyword'] = r_keyword.strip().lower()
                        df_jobs.to_csv(path)
                trans_keyword_list.append(search_keyword)

                for (
                    fem_keyword,
                    male_keyword,
                    gen_keyword,
                    old_keyword,
                    young_keyword,
                    age_keyword,
                ) in itertools.zip_longest(
                    args['keywords_womenvocc'],
                    args['keywords_menvocc'],
                    args['keywords_genvsect'],
                    args['keywords_oldvocc'],
                    args['keywords_youngvocc'],
                    args['keywords_agevsect'],
                ):
                    if search_keyword == str(fem_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Female'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Gender'].append(str(row['Job ID']))
                    if search_keyword == str(male_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Male'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Gender'].append(str(row['Job ID']))

                    if search_keyword == str(old_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Older Worker'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Age'].append(str(row['Job ID']))
                    if search_keyword == str(young_keyword).strip().lower().replace("-Noon's MacBook Pro",''):
                        job_id_dict['Younger Worker'].append(str(row['Job ID']))
                    else:
                        job_id_dict['Mixed Age'].append(str(row['Job ID']))

    trans_keyword_list = save_trans_keyword_list(trans_keyword_list)
    if args['save_enabled'] is True:
        with open(f'{args["parent_dir"]}{all_save_path}.json', 'w', encoding='utf8') as f:
            json.dump(job_id_dict, f)

    elif args['save_enabled'] is False:
        print('No job id matching save enabled.')

    return job_id_dict


# %%
# Function to split job ads to sentences
def split_df_jobs_to_df_sent(
    df_for_analysis,
    lst_col='Job Description',
    pattern=r'[\n\r]+|(?<=[a-z]\.)(?=\s*[A-Z])|(?<=[a-z])(?=[A-Z])',
    args=get_args(),
):
    dff = df_for_analysis.assign(
        **{
            lst_col: df_for_analysis[lst_col]
            .swifter.progress_bar(args['print_enabled'])
            .apply(lambda x: sent_tokenize(x))
        }
    )
    df_final = pd.DataFrame(
        {
            col: np.repeat(dff[col].values, dff[lst_col].str.len())
            for col in dff.columns.difference([lst_col])
        }
    ).assign(**{lst_col: np.concatenate(dff[lst_col].values)})[dff.columns.to_list()]

    return df_final


# %%
# Function to split job descriptions to sentences
def split_to_sentences(df_jobs, df_sentence_list=None, args=get_args()):
    print('-' * 20)
    print(
        f'NOTE: The function "get_args" which is used by the "split_to_sentences" function contains the following optional (default) arguments:\n{get_default_args(get_args)}\nYou can change these arguments by calling "get_args",  passing the desired variable values to "get_args" then passing "get_args" to "split_to_sentences".'
    )
    print('-' * 20)

    if df_sentence_list is None:
        df_sentence_list = []

    if isinstance(df_jobs, list):
        df_list = df_jobs
        if args['print_enabled'] is True:
            print(f'LIST OF {len(df_list)} DFs passed.')
            print('-' * 20)
        for df_jobs in df_list:
            if isinstance(df_jobs, pd.DataFrame):
                if (
                    (not df_jobs.empty)
                    and all(df_jobs['Language'] == str(args['language']))
                    and (df_jobs is not None)
                ):
                    if args['print_enabled'] is True:
                        print(f'DF OF LENGTH {len(df_jobs)} passed.')
                    try:
                        if args['print_enabled'] is True:
                            print(
                                f'Processing DF from platform: {df_jobs["Platform"].iloc[0]}'
                            )
                        (
                            search_keyword,
                            job_id,
                            age,
                            args,
                            sentence_list,
                            sentence_dict,
                            df_sentence,
                            df_sentence_all,
                        ) = split_to_sentences_helper(df_jobs, args)
                        df_sentence_list.append(df_sentence)
                        if args['txt_save'] is True:
                            if args['print_enabled'] is True:
                                print(
                                    f'Saving {df_jobs["Search Keyword"].iloc[0]} DF to txt.'
                                )
                            write_all_to_txt(search_keyword, job_id, age, df_jobs, args)
                        elif args['txt_save'] is False:
                            if args['print_enabled'] is True:
                                print(
                                    f'No txt save enabled for DF {df_jobs["Search Keyword"].iloc[0]}.'
                                )
                    except Exception:
                        pass
                elif (
                    (df_jobs.empty)
                    or all(df_jobs['Language'] != str(args['language']))
                    or (df_jobs is None)
                ):
                    if df_jobs.empty:
                        if args['print_enabled'] is True:
                            print('DF is empty.')
                    elif all(df_jobs['Language'] != str(args['language'])):
                        if args['print_enabled'] is True:
                            print(
                                f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                            )
            elif isinstance(df_jobs, list):
                df_sentence, df_sentence_list = split_to_sentences(df_jobs)
                df_sentence_list.append(df_sentence)
    #         pbar.finish()
    elif isinstance(df_jobs, pd.DataFrame):
        if (
            (not df_jobs.empty)
            and all(df_jobs['Language'] == str(args['language']))
            and (df_jobs is not None)
        ):
            if args['print_enabled'] is True:
                print(f'DF OF LENGTH {len(df_jobs)} passed.')
            try:
                if args['print_enabled'] is True:
                    print(f'Processing DF from platform: {df_jobs["Platform"].iloc[0]}')
                    (
                        search_keyword,
                        job_id,
                        age,
                        args,
                        sentence_list,
                        sentence_dict,
                        df_sentence,
                        df_sentence_all,
                    ) = split_to_sentences_helper(df_jobs, args)
                df_sentence_list.append(df_sentence)
                if args['txt_save'] is True:
                    if args['print_enabled'] is True:
                        print(f'Saving {df_jobs["Search Keyword"].iloc[0]} DF to txt.')
                    write_all_to_txt(search_keyword, job_id, age, df_jobs, args)
                elif args['txt_save'] is False:
                    if args['print_enabled'] is True:
                        print(
                            f'No txt save enabled for DF {df_jobs["Search Keyword"].iloc[0]}.'
                        )
            except Exception:
                pass
        elif (
            (df_jobs.empty)
            or all(df_jobs['Language'] != str(args['language']))
            or (df_jobs is None)
        ):
            if df_jobs.empty:
                if args['print_enabled'] is True:
                    print('DF is empty.')
            elif all(df_jobs['Language'] != str(args['language'])):
                if args['print_enabled'] is True:
                    print(
                        f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                    )

    try:
        if (not df_sentence.empty) and (args['save_enabled'] is True):
            df_sentence_all.to_pickle(args['parent_dir'] + f'df_sentence_all_jobs.{args["file_save_format"]}')
            # pickle.dump(df_sentence_all, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif args['save_enabled'] is False:
            if args['print_enabled'] is True:
                print('No sentence save enabled.')
    except Exception:
        df_sentence = pd.DataFrame()
        if args['print_enabled'] is True:
            print('No sentence df found.')

    return df_sentence, df_sentence_list


# %%
def split_to_sentences_helper(df_jobs, args=get_args()):
    if (not df_jobs.empty) and (len(df_jobs != 0)):
        if args['print_enabled'] is True:
            print(
                f'DF {str(df_jobs["Search Keyword"].iloc[0])} of length {df_jobs.shape[0]} passed.'
            )
        try:
            search_keyword = '_'.join(
                str(df_jobs['Search Keyword'].iloc[0]).lower().split(' ').replace("-Noon's MacBook Pro",'')
            )
            if (df_jobs['Job ID'] == df_jobs['Job ID'].iloc[0]).all():
                job_id = str(df_jobs['Job ID'].iloc[0])
                age = str(df_jobs['Age'].iloc[0])
                (
                    sentence_list,
                    sentence_dict,
                    df_sentence,
                    df_sentence_all,
                ) = sent_tokenize_and_save_df(search_keyword, job_id, age, df_jobs, args)
            else:
                job_ids = list(df_jobs['Job ID'].unique())
                ages = list(df_jobs['Age'].unique())
                for job_id in job_ids:
                    for age in ages:
                        (
                            sentence_list,
                            sentence_dict,
                            df_sentence,
                            df_sentence_all,
                        ) = sent_tokenize_and_save_df(search_keyword, job_id, age, df_jobs, args)

                        yield (
                            search_keyword,
                            job_id,
                            age,
                            args,
                            sentence_list,
                            sentence_dict,
                            df_sentence,
                            df_sentence_all,
                        )

        except Exception as e:
            if args['print_enabled'] is True:
                print(e.json())
            (
                search_keyword,
                job_id,
                age,
                sentence_list,
                sentence_dict,
                df_sentence,
                df_sentence_all,
            ) = assign_all(7, None)
    elif df_jobs.empty:
        (
            search_keyword,
            job_id,
            age,
            sentence_list,
            sentence_dict,
            df_sentence,
            df_sentence_all,
        ) = assign_all(7, None)

    return (
        search_keyword,
        job_id,
        age,
        args,
        sentence_list,
        sentence_dict,
        df_sentence,
        df_sentence_all,
    )


# %%
# Function to tokenize and clean job descriptions from df based on language
def sent_tokenize_and_save_df(search_keyword, job_id, age, df_jobs, args=get_args()):
    if (not df_jobs.empty) and all(df_jobs['Language'] == str(args['language'])):
        path_to_csv = str(
            args['parent_dir']
            + f'Sentences DF/{str(args["language"])}/{age}/{str(" ".join(search_keyword.split("_")))}'
        )
        pathlib.Path(path_to_csv).mkdir(parents=True, exist_ok=True)

        lang_num = df_jobs.loc[
            df_jobs.Language == str(args['language']), 'Language'
        ].count()
        if args['print_enabled'] is True:
            print(f'{lang_num} jobs with language {str(args["language"])} found.')
        if lang_num > 0:
            if args['print_enabled'] is True:
                print(
                    f'Tokenizing DF {str(" ".join(search_keyword.split("_")))} of length {df_jobs.shape[0]} to sentences.'
                )
            sentence_dict = {}
            for index, row in df_jobs.iterrows():
                pattern = r'[\n\r]+|(?<=[a-z]\.)(?=\s*[A-Z])|(?<=[a-z])(?=[A-Z])'
                # sentence_list = []
                if row.loc['Language'] == str(args['language']):
                    sentence_list = [re.split(pattern, sent) for sent in list(sent_tokenize(row['Job Description']))]
                    sentence_list = [re.split(pattern, sent) for sent in list(nlp(row['Job Description']).sents)]
                    sentence_dict[str(row.loc['Job ID'])] = list(sentence_list)
                    sentence_dict['Search Keyword'] = row['Search Keyword']
                    sentence_dict['Gender'] = row['Gender']
                    sentence_dict['Age'] = row['Age']

            # Create DF sentence from sentence dict
            df_sentence_all = pd.DataFrame()
            for key, lst in sentence_dict.items():
                # if (key != "Search Keyword") and (key != "Gender") and (key != "Age"):
                df_sentence = pd.DataFrame(
                    [(key, sent) for sent in lst], columns=args['columns_fill_list']
                )
                df_sentence = df_sentence.reindex(
                    columns=[
                        *df_sentence.columns.to_list(),
                        *[
                            col
                            for col in args['columns_list']
                            if col not in args['columns_fill_list']
                        ],
                    ],
                    fill_value=0,
                )
                df_sentence_all = pd.concat([df_sentence, df_sentence_all])

                if not df_sentence.empty:
                    if args['print_enabled'] is True:
                        print(
                            f'Saving sentences DF {sentence_dict["Search Keyword"]} of length {df_sentence.shape[0]} and job ID {df_sentence["Job ID"].iloc[0]} to csv.'
                        )
                    df_sentence.to_csv(
                        path_to_csv
                        + f'/Job ID - {df_sentence["Job ID"].iloc[0]}_sentences_df.csv',
                        mode='w',
                        sep=',',
                        header=True,
                        index=True,
                    )
                    # Write DF to excel
                    if args['excel_save'] is True:
                        if args['print_enabled'] is True:
                            print(
                                f'Saving {df_jobs["Search Keyword"].iloc[0]} DF to excel.'
                            )
                        write_sentences_to_excel(
                            search_keyword, job_id, age, df_sentence, args
                        )
                elif df_sentence.empty:
                    if args['print_enabled'] is True:
                        print('Sentence DF is empty.')
                    (
                        sentence_list,
                        sentence_dict,
                        df_sentence,
                        df_sentence_all,
                    ) = assign_all(4, None)

            # Create DF which inclues Search Keyword and Age
            df_sentence_all['Search Keyword'] = sentence_dict['Search Keyword']
            df_sentence_all['Gender'] = sentence_dict['Gender']
            df_sentence_all['Age'] = sentence_dict['Age']
            df_sentence_all = df_sentence_all[
                ['Search Keyword', 'Gender', 'Age']
                + [
                    col
                    for col in df_sentence_all.columns
                    if col not in ['Search Keyword', 'Gender', 'Age']
                ]
            ]

            if not df_sentence_all.empty:
                if args['print_enabled'] is True:
                    print(
                        f'Saving ALL sentences DF {sentence_dict["Search Keyword"]} of length {df_sentence_all.shape[0]} and job ID {df_sentence_all["Job ID"].iloc[0]} to csv.'
                    )
                df_sentence_all.to_csv(
                    path_to_csv + f'/ALL_{search_keyword}_sentences_df.{args["file_save_format_backup"]}',
                    mode='w',
                    sep=',',
                    header=True,
                    index=True,
                )
            elif df_sentence_all.empty:
                if args['print_enabled'] is True:
                    print('ALL sentence DF is empty.')
                sentence_list, sentence_dict, df_sentence, df_sentence_all = assign_all(
                    4, None
                )

        elif lang_num <= 0:
            if args['print_enabled'] is True:
                print(f'No {str(args["language"])} language jobs found.')
            sentence_list, sentence_dict, df_sentence, df_sentence_all = assign_all(
                4, None
            )

    elif (df_jobs.empty) or all(df_jobs['Language'] != str(args['language'])):
        if df_jobs.empty:
            if args['print_enabled'] is True:
                print('DF is empty.')
        elif all(df_jobs['Language'] != str(args['language'])):
            if args['print_enabled'] is True:
                print(
                    f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                )
        sentence_list, sentence_dict, df_sentence, df_sentence_all = assign_all(4, None)

    return sentence_list, sentence_dict, df_sentence, df_sentence_all


# %%
# Function to save individual sentences in excel file
def write_sentences_to_excel(search_keyword, job_id, age, df_sentence, args=get_args()):
    if (df_sentence is not None) and (not df_sentence.empty):
        path_to_txt = str(
            args['parent_dir']
            + f'Jobs EXECL/{str(args["language"])}/{age}/{str(" ".join(search_keyword.split("_")))}'
        )
        pathlib.Path(path_to_txt).mkdir(parents=True, exist_ok=True)
        # Create column dict for excel file
        column_dict = [{'header': str(col)} for col in args['columns_list']]

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(
            path_to_txt
            + f'/Job ID - {df_sentence["Job ID"].iloc[0]}_Coder_Name - Codebook (Automating Equity).xlsx',
            engine='xlsxwriter',
        )

        # Check datatype
        if isinstance(df_sentence, pd.DataFrame):
            df_sentence, workbook, worksheet = write_sentences_to_excel_helper(
                1, writer, df_sentence, args
            )
        elif isinstance(df_sentence, list):
            df_list = df_sentence
            for i, df in enumerate(df_list, 1):
                if isinstance(df, pd.DataFrame):
                    df_sentence = df
                    df_sentence, workbook, worksheet = write_sentences_to_excel_helper(
                        i, writer, df_sentence, args
                    )

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        try:
            workbook.close()
        except Exception:
            pass


# %%
def write_sentences_to_excel_helper(i, writer, df_sentence, args=get_args()):
    try:
        # Convert the dataframe to an XlsxWriter Excel object.
        df_sentence.to_excel(writer, sheet_name=f'Sheet {i}')

        # Get the xlsxwriter objects from the dataframe writer object.
        workbook = writer.book
        worksheet = writer.sheets[f'Sheet {i}']

        # Add a format for the header cells.
        header_format = workbook.add_format(args['format_props'])

        # Add a format for columns
        first_row = 1  # Excluding header
        first_col = 3  # Excluding index, Job ID and Sentence
        last_row = len(df_sentence)
        last_col = len(args['columns_list'])
        worksheet.data_validation(
            first_row, first_col, last_row, last_col, args['validation_props']
        )

    except Exception as e:
        if args['print_enabled'] is True:
            print(e.json())

    return (df_sentence, workbook, worksheet)


# %%
# Function to save full job description text in txt file
def write_all_to_txt(search_keyword, job_id, age, df_jobs, args=get_args()):

    if isinstance(df_jobs, list):
        df_list = df_jobs
        for df_jobs in df_list:
            if isinstance(df_jobs, pd.DataFrame):
                if (not df_jobs.empty) and all(
                    df_jobs['Language'] == str(args['language'])
                ):
                    try:
                        write_all_to_txt_helper(
                            search_keyword, job_id, age, df_jobs, args
                        )
                    except Exception as e:
                        if args['print_enabled'] is True:
                            print(e.json())
                elif (df_jobs.empty) or all(
                    df_jobs['Language'] != str(args['language'])
                ):
                    if df_jobs.empty:
                        if args['print_enabled'] is True:
                            print('DF is empty.')
                    elif all(df_jobs['Language'] != str(args['language'])):
                        if args['print_enabled'] is True:
                            print(
                                f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                            )
            elif isinstance(df_jobs, list):
                write_all_to_txt(search_keyword, job_id, age, df_jobs, args)
    #         pbar.finish()
    elif isinstance(df_jobs, pd.DataFrame):
        if (not df_jobs.empty) and all(df_jobs['Language'] == str(args['language'])):
            write_all_to_txt_helper(search_keyword, job_id, age, df_jobs, args)
        elif (df_jobs.empty) or all(df_jobs['Language'] != str(args['language'])):
            if df_jobs.empty:
                if args['print_enabled'] is True:
                    print('DF is empty.')
            elif all(df_jobs['Language'] != str(args['language'])):
                if args['print_enabled'] is True:
                    print(
                        f'No valid language found in {df_jobs["Search Keyword"].iloc[0]} DF.'
                    )


# %%
def write_all_to_txt_helper(search_keyword, job_id, age, df_jobs, args=get_args()):
    path_to_txt = (
        str(args['parent_dir'])
        + f'Jobs TXT/{args["language"]}/{age}/{" ".join(df_jobs["Search Keyword"].iloc[0].split("_"))}'
    )
    pathlib.Path(path_to_txt).mkdir(parents=True, exist_ok=True)

    df_jobs.drop(
        [x for x in args['columns_drop_list'] if x in df_jobs.columns], axis=1, inplace=True,
    )
    df_jobs.drop(
        df_jobs.columns[df_jobs.columns.str.contains('Age', case=False)],
        axis=1,
        inplace=True,
    )

    for index, row in df_jobs.iterrows():
        if row['Language'] == str(args['language']):
            with open(path_to_txt + f'/Job ID - {str(row["Job ID"])}.txt', 'a') as f:
                f.write(row['Job Description'])


# %%
# Function to send batches of excel files to google drive
def send_new_excel_to_gdrive(
    files_to_upload_number=20,
    coders_dict='',
    coders_from_dict=True,
    language='en',
    move_txt_file=True,
    gender_list=['Female', 'Male', 'Mixed Gender'],
    done_job_excel_list=None,
    new_job_excel_list=None,
    new_batch_job_txt_list=None,
    args=get_args(),
):
    dest_path=validate_path(f'{args["content_analysis_dir"]}')

    print('-' * 20)
    print(
        f'NOTE: The function "send_new_excel_to_gdrive" contains the following optional (default) arguments:\n{get_default_args(send_new_excel_to_gdrive)}.'
    )
    print('-' * 20)

    if done_job_excel_list is None:
        done_job_excel_list = []
    if new_job_excel_list is None:
        new_job_excel_list = []
    if new_batch_job_txt_list is None:
        new_batch_job_txt_list = []

    if coders_from_dict is True:
        with open(f'{args["parent_dir"]}coders_dict.json', encoding='utf8') as f:
            coders_dict = json.load(f)
    elif coders_from_dict is False:
        pass
    with open(f'{args["parent_dir"]}batch_counter_dict.json', encoding='utf8') as f:
        batch_counter_dict = json.load(f)

    for coder_number, coder_name in coders_dict.items():
        coder_dest_folder = validate_path(dest_path + f'{coder_name} Folder/')
        if os.path.isdir(coder_dest_folder):
            for coder_dest_folder_path, batch_folder_names, done_job_excel in os.walk(
                coder_dest_folder
            ):
                for batch_number in batch_folder_names:
                    batch_counter_dict[coder_name].extend(
                        int(i)
                        for i in re.findall(r'\d+', batch_number)
                        if int(i) not in batch_counter_dict[coder_name]
                    )
                for done_job_excel_name in done_job_excel:
                    if ('Job ID - ') and ('.xlsx') in done_job_excel_name:
                        if is_non_zero_file(coder_dest_folder_path + '/' + done_job_excel_name) is True:
                            done_job_excel_list.append(
                                validate_path(
                                    coder_dest_folder_path + '/' + done_job_excel_name
                                )
                            )

        try:
            batch_counter_dict[coder_name] = list(set(batch_counter_dict[coder_name]))
            done_job_excel_list = list(set(done_job_excel_list))
        except Exception:
            pass

        excel_source_folder = validate_path(
            f'{args["parent_dir"]}Jobs EXECL/{str(language)}'
        )
        if os.path.isdir(excel_source_folder):
            for (
                gender_occ_source_dir_path,
                all_dir_file_names,
                new_job_excel,
            ) in os.walk(excel_source_folder):
                for new_job_excel_name in new_job_excel:
                    if ('Job ID - ') and ('.xlsx') in new_job_excel_name:
                        if (is_non_zero_file(gender_occ_source_dir_path + '/' + new_job_excel_name) is True
                            and (
                                new_job_excel
                                != any(
                                    done_job_excel
                                    for done_job_excel in done_job_excel_list
                                )
                            )
                            and ('.DS_Store' not in new_job_excel)
                        ):
                            new_job_excel_list.append(
                                validate_path(
                                    gender_occ_source_dir_path
                                    + '/'
                                    + new_job_excel_name
                                )
                            )
        new_job_excel_list = list(set(new_job_excel_list))

        if len(new_job_excel_list) > int(files_to_upload_number):
            new_batch_job_excel_list = random.sample(
                new_job_excel_list, int(files_to_upload_number)
            )
        elif len(new_job_excel_list) <= int(files_to_upload_number):
            new_batch_job_excel_list = new_job_excel_list
            if args['print_enabled'] is True:
                print(
                    f'Less than 12 excel jobs remaining. Moving final {len(new_batch_job_excel_list)} jobs.'
                )

        if move_txt_file is True:
            for new_batch_job_excel in new_batch_job_excel_list:
                new_batch_job_txt_list.append(
                    str(new_batch_job_excel)
                    .replace('Jobs EXECL', 'Jobs TXT')
                    .replace('_Coder_Name - Codebook (Automating Equity).xlsx', '.txt')
                )
            new_batch_job_txt_list = list(set(new_batch_job_txt_list))

        if len(new_batch_job_excel_list) > 0:
            if os.path.isdir(coder_dest_folder):
                for (
                    coder_dest_folder_path,
                    batch_folder_names,
                    done_job_excel,
                ) in os.walk(coder_dest_folder):
                    path_to_next_batch = coder_dest_folder + str(
                        f'{coder_name} Folder - Batch {max(int(i) for v in batch_counter_dict.values() for i in v) + 1}/'
                    )
                    pathlib.Path(path_to_next_batch).mkdir(parents=True, exist_ok=True)
                    for (
                        new_batch_job_excel,
                        new_batch_job_txt,
                    ) in itertools.zip_longest(
                        new_batch_job_excel_list, new_batch_job_txt_list
                    ):
                        try:
                            shutil.move(new_batch_job_excel, path_to_next_batch)
                            if move_txt_file is True:
                                shutil.move(new_batch_job_txt, path_to_next_batch)
                        except Exception:
                            pass
        elif len(new_batch_job_excel_list) <= 0:
            if args['print_enabled'] is True:
                print('No more files to move.')

    for coder_number, coder_name in list(coders_dict.items()):
        for coder_dest_folder_path, batch_folder_names, done_job_excel in os.walk(
            coder_dest_folder
        ):
            for batch_number in batch_folder_names:
                batch_counter_dict[coder_name].extend(
                    int(i)
                    for i in re.findall(r'\d+', batch_number)
                    if i not in batch_counter_dict[coder_name]
                )
        try:
            batch_counter_dict[coder_name] = list(set(batch_counter_dict[coder_name]))
        except Exception:
            pass

    if args['save_enabled'] is True:
        with open(f'{args["parent_dir"]}batch_counter_dict.json', 'w', encoding='utf8') as f:
            json.dump(batch_counter_dict, f)
    elif args['save_enabled'] is False:
        if args['print_enabled'] is True:
            print('No batch counter save enabled.')


# %% [markdown]
# Analyses

# %%
# Function to plot
def qq_plot(x):
    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(x, dist='norm', plot=None)
    plt.plot(osm, osr, '.', osm, slope * osm + intercept)
    plt.xlabel('Quantiles', fontsize=14)
    plt.ylabel('Quantiles Obs', fontsize=14)
    plt.show()


# %%
# Function to create dummy variables
def encodeY(Y):

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y


# Or use LabelBinarizer()

# %%
# Function to find cosine similarity score: 1−cos(x, y) = (x * y)/(||x||*||y||)
def cosine_similarity(x, y):
    x_sqrt = np.sqrt(np.dot(x, x))
    y_sqrt = np.sqrt(np.dot(y, y))
    if y_sqrt != 0:
        return np.dot(x, y.T) / (x_sqrt * y_sqrt)
    elif y_sqrt == 0:
        return 0


# %%
# Function to open and clean dfs
def open_and_clean_excel(
    EXCEL_PATHS=defaultdict(list),
    front_columns=['Coder ID', 'Job ID', 'OG_Sentence ID', 'Sentence ID', 'Sentence'],
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    int_variable: str = 'Job ID',
    str_variable='Sentence',
    reset=True,
    args=get_args(),
):

    dest_path=validate_path(f'{args["content_analysis_dir"]}')

    for lst in EXCEL_PATHS.values():
        lst[:] = list(set(lst))
    if len(EXCEL_PATHS) < 5:
        if args['print_enabled'] is True:
            print(
                f'NOTE: The function "open_and_clean_excel" contains the following optional (default) arguments:\n{get_default_args(open_and_clean_excel)}'
            )
    with open(f'{args["parent_dir"]}coders_dict.json', encoding='utf8') as f:
        coders_dict = json.load(f)
    for coder_number, coder_name in coders_dict.items():
        coder_dest_folder = validate_path(f'{dest_path}{coder_name} Folder/')
        if os.path.isdir(coder_dest_folder):
            for coder_dest_folder_path, batch_folder_names, done_job_excel in os.walk(
                coder_dest_folder
            ):
                for done_job_excel_name in done_job_excel:
                    if (
                        (len(done_job_excel) != 0)
                        and ('Job ID - ' in done_job_excel_name)
                        and ('.xlsx' in done_job_excel_name)
                        and ('.txt' not in done_job_excel_name)
                        and (is_non_zero_file(coder_dest_folder_path + '/' + done_job_excel_name) is True)
                    ):
                        EXCEL_PATHS[coder_name].append(
                            validate_path(
                                coder_dest_folder_path + '/' + done_job_excel_name
                            )
                        )
                    elif (
                        (len(done_job_excel) == 0)
                        or (
                            ('Job ID - ' not in done_job_excel_name)
                            and ('.xlsx' not in done_job_excel_name)
                            and ('.txt' in done_job_excel_name)
                        )
                        and (
                            is_non_zero_file(coder_dest_folder_path + '/' + done_job_excel_name) is False
                        )
                    ):
                        coders_dict.pop(coder_number, None)
        if args['print_enabled'] is True:
            print(
                f'{len(EXCEL_PATHS[coder_name])} valid excel files found for coder {coder_name}.'
            )

    if len(EXCEL_PATHS) == 0:
        if args['print_enabled'] is True:
            print('No valid excel files found for any coders.')
        (
            coders_list,
            coders_dict,
            coders_numbers,
            df_coder_list,
            df_concat_coder_all,
        ) = assign_all(5, None)
    elif len(EXCEL_PATHS) != 0:
        if args['print_enabled'] is True:
            print('-' * 20)
        coders_list = list(coders_dict.values())
        coders_numbers = list(coders_dict.keys())
        df_coder_list = []
        for index, (coder_key, CODER_EXCEL_PATH) in enumerate(EXCEL_PATHS.items()):
            for path in CODER_EXCEL_PATH:
                if args['print_enabled'] is True:
                    print(path)
                # file_extension = path.lower().split('.')[-1]
                if path.endswith('xlsx'):
                    df_coder = pd.read_excel(
                        validate_path(path), index_col=0, engine='openpyxl'
                    )
                elif path.endswith('xls'):
                    df_coder = pd.read_excel(validate_path(path), index_col=0)
                else:
                    raise Exception('File not supported')

                if df_coder.columns.str.contains('^Unnamed').all():
                    break
                else:
                    df_coder = clean_df(df_coder, str_variable=str_variable, reset=reset)
                    df_coder.drop(
                        df_coder.columns[
                            df_coder.columns.str.contains('Coder Remarks', case=False)
                        ],
                        axis=1,
                        inplace=True,
                    )
                    df_coder['Job ID'].fillna(method='ffill', inplace=True)
                    for k, v in coders_dict.items():
                        if v == coder_key:
                            df_coder['Coder ID'] = k
                    df_coder[f'OG_{str_variable} ID'] = df_coder.index + 1
                    df_coder = df_coder.fillna(0)
                    df_coder[str_variable] = df_coder[str_variable].apply(lambda sentence: sentence.strip().lower().replace('[^\w\s]', ''))
                    if df_coder[str_variable].isna().sum() > 0:
                        if args['print_enabled'] is True:
                            print(
                                f'{df_coder[str_variable].isna().sum()} missing sentences found.'
                            )
                    df_coder_list.append(df_coder)

        # pbar.finish()

        if len(df_coder_list) >= 1:
            df_concat_coder_all = pd.concat(df_coder_list)
            df_concat_coder_all[f'{str_variable} ID'] = (
                df_concat_coder_all.groupby([str_variable]).ngroup() + 1
            )
            df_concat_coder_all = df_concat_coder_all[
                front_columns
                + [
                    col
                    for col in df_concat_coder_all.columns
                    if col not in front_columns
                ]
            ]
            df_concat_coder_all.loc[:, cal_columns] = (
                df_concat_coder_all.loc[:, cal_columns]
                .swifter.progress_bar(args['print_enabled'])
                .apply(pd.to_numeric, downcast='integer', errors='coerce')
            )
            df_concat_coder_all = clean_df(df_concat_coder_all, str_variable=str_variable, reset=reset)
            df_concat_coder_all.index = range(df_concat_coder_all.shape[0])
            if args['print_enabled'] is True:
                print(f'Total of {len(df_concat_coder_all)} sentences in the dataset.')
            for var in cal_columns:
                if (df_concat_coder_all[str(var)] == 1).sum() + (
                    df_concat_coder_all[str(var)] == 0
                ).sum() == len(df_concat_coder_all):
                    if args['print_enabled'] is True:
                        print(
                            f'Sum of "present" and "not present" {str(var)} labels is equal to length of dataset.'
                        )
                else:
                    if args['print_enabled'] is True:
                        print(
                            f'Sum of "present" and "not present" {str(var)} labels is NOT equal to length of dataset.'
                        )
                    raise ValueError('Problem with candidate trait labels count.')
        elif len(df_coder_list) <= 1:
            df_concat_coder_all = df_coder_list

    if args['print_enabled'] is True:
        print('-' * 20)

    return (
        coders_list,
        coders_dict,
        coders_numbers,
        df_coder_list,
        df_concat_coder_all,
    )


# %% [markdown]
# intercoder_reliability_to_csv

# %%
# Calculate k-alpha
def IR_kalpha(
    df_concat_coder_all,
    save_enabled=False,
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    k_alpha_dict=None,
    args=get_args(),
):
    print(
        f'NOTE: The function "IR_kalpha" contains the following optional (default) arguments:\n{get_default_args(IR_kalpha)}'
    )

    if k_alpha_dict is None:
        k_alpha_dict = {}

    for column in df_concat_coder_all[cal_columns]:
        k_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
            df_concat_coder_all,
            experiment_col='Sentence ID',
            annotator_col='Coder ID',
            class_col=str(column),
        )
        print('-' * 20)
        k_alpha_dict['K-alpha ' + str(column)] = k_alpha
        print(f"Krippendorff's alpha ({str(column)}): ", k_alpha)
    print('-' * 20)

    if save_enabled is True:
        with open(f'{args["parent_dir"]}K-alpha.json', 'w', encoding='utf8') as f:
            json.dump(k_alpha_dict, f)
    elif save_enabled is False:
        print('No K-alpha save enabled.')

    return k_alpha_dict


# %%
# Calculate all IR
def IR_all(
    df_concat_coder_all,
    coders_numbers,
    save_enabled=True,
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    coder_score_dict=defaultdict(list),
    ir_all_dict=None,
    args=get_args(),
):
    print(
        f'NOTE: The function "IR_all" contains the following optional (default) arguments:\n{get_default_args(IR_all)}'
    )

    if ir_all_dict is None:
        ir_all_dict = {}

    for column in df_concat_coder_all[cal_columns]:
        for index, coder_number in enumerate(coders_numbers):
            coder = (
                df_concat_coder_all.loc[
                    df_concat_coder_all['Coder ID'].astype(str) == str(coder_number), str(column)
                ]
                .to_list()
            )
            coder_score_dict[str(column)].append(
                [[int(index + 1), i, coder[i]] for i in range(len(coder))]
            )

    for column in cal_columns:
        counter = 0
        formatted_codes = coder_score_dict[str(column)][counter]
        while counter < len(coders_numbers) - 1:
            try:
                formatted_codes += coder_score_dict[str(column)][counter + 1]
                counter += 1
            except Exception:
                break

        ratingtask = agreement.AnnotationTask(data=formatted_codes)

        ir_all_dict['IR K-alpha ' + str(column)] = ratingtask.alpha()
        ir_all_dict['IR Cohen-kappa ' + str(column)] = ratingtask.kappa()
        ir_all_dict['IR Scott-pi ' + str(column)] = ratingtask.pi()

        print('-' * 20, '\n')
        print(f"Krippendorff's alpha ({str(column)}):", ratingtask.alpha())
        print(f"Cohen's Kappa ({str(column)}): ", ratingtask.kappa())
        print(f"Scott's pi ({str(column)}): ", ratingtask.pi())
    print('-' * 20, '\n')

    if save_enabled is True:
        with open(f'{args["parent_dir"]}IR_all.json', 'w', encoding='utf8') as f:
            json.dump(ir_all_dict, f)
    elif save_enabled is False:
        print('No IR save enabled.')

    return (coder_score_dict, ratingtask, ir_all_dict)


# %%
def IR_all_final(
    coder,
    k_alpha_dict=None,
    ir_all_dict=None,
    coders_numbers=[1, 2],
    coder_score_dict=defaultdict(list),
    save_enabled=True,
    front_columns=['Coder ID', 'Job ID', 'OG_Sentence ID', 'Sentence ID', 'Sentence'],
    cal_columns=[
        'Warmth',
        'Competence',
        'Task_Mentioned',
        'Task_Warmth',
        'Task_Competence',
    ],
    args=get_args(),
):
    if k_alpha_dict is None:
        k_alpha_dict = {}
    if ir_all_dict is None:
        ir_all_dict = {}

    reliability_dir=f'{args["content_analysis_dir"]}Reliability Checks/'
    print('-' * 20)
    print('\n')
    print(f'Results for {ir_file_name}')
    print('-' * 20)
    print('\n')
    ############# INTRACODER Coder 1 #############
    if coder == 1:
        ir_file_name = 'INTRACODER1'
        df1a = pd.read_excel(
            f'{reliability_dir}Pair 1 - Intra/Job ID - p_ce05575325f3b0f1_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1b = pd.read_excel(
            f'{reliability_dir}Pair 2 - Intra/Job ID - p_ca008a8d67189539_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1c = pd.read_excel(
            f'{reliability_dir}Pair 3 - Intra/Job ID - p_9acfa03a05f2542f_Rhea- Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1d = pd.read_excel(
            f'{reliability_dir}Pair 4 - Intra/Job ID - p_3d626cbfef055cb4_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1e = pd.read_excel(
            f'{reliability_dir}Pair 5 - Intra/Job ID - p_1b37ad5237066811_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )

        df1 = pd.concat([df1a, df1b, df1c, df1d, df1e])

        df2a = pd.read_excel(
            f'{reliability_dir}Pair 1 - Intra/OLD Job ID - p_ce05575325f3b0f1_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2b = pd.read_excel(
            f'{reliability_dir}Pair 2 - Intra/OLD Job ID - p_ca008a8d67189539_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2c = pd.read_excel(
            f'{reliability_dir}Pair 3 - Intra/OLD Job ID - p_9acfa03a05f2542f_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2d = pd.read_excel(
            f'{reliability_dir}Pair 4 - Intra/OLD Job ID - p_3d626cbfef055cb4_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2e = pd.read_excel(
            f'{reliability_dir}Pair 5 - Intra/OLD Job ID - p_1b37ad5237066811_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )

        df2 = pd.concat([df2a, df2b, df2c, df2d, df2e])

    ############# INTRACODER Coder 2 #############
    elif coder == 2:
        ir_file_name = 'INTRACODER2'
        df1a = pd.read_excel(
            f'{reliability_dir}Pair 8 Intra/OLD Job ID - p_a087b464a6a092fa_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1b = pd.read_excel(
            f'{reliability_dir}Pair 11 Intra/Job ID - 4052472440_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1c = pd.read_excel(
            f'{reliability_dir}Pair 12 Intra/Job ID - p_7674c23f38f94dcf_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1d = pd.read_excel(
            f'{reliability_dir}Pair 13 Intra/Job ID - p_42ea0a6f52e862d4_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df1e = pd.read_excel(
            f'{reliability_dir}Pair 14 Intra/Job ID - p_9f364da9030d1ce6_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )

        df1 = pd.concat([df1a, df1b, df1c, df1d, df1e])

        df2a = pd.read_excel(
            f'{reliability_dir}Pair 8 Intra/PAIRED Job ID - p_a087b464a6a092fa_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2b = pd.read_excel(
            f'{reliability_dir}Pair 11 Intra/PAIRED Job ID - 4052472440_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2c = pd.read_excel(
            f'{reliability_dir}Pair 12 Intra/PAIRED Job ID - p_7674c23f38f94dcf_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2d = pd.read_excel(
            f'{reliability_dir}Pair 13 Intra/PAIRED Job ID - p_42ea0a6f52e862d4_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )
        df2e = pd.read_excel(
            f'{reliability_dir}Pair 14 Intra/PAIRED Job ID - p_9f364da9030d1ce6_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )

        df2 = pd.concat([df2a, df2b, df2c, df2d, df2e])

    ############# INTERCODER #############
    elif coder == 'all':
        ir_file_name = 'INTERCODER'

        df1 = pd.read_excel(
            f'{reliability_dir}Pair 6 - Inter/PAIRED INTER - Job ID - p_15a42cd4b082799e_Rhea - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )

        df2 = pd.read_excel(
            f'{reliability_dir}Pair 6 - Inter/OLD Job ID - p_15a42cd4b082799e_Coder_Name - Codebook (Automating Equity).xlsx',
            index_col=0,
            engine='openpyxl',
        )

    df1['Coder ID'] = 1
    df1['OG_Sentence ID'] = df1.index + 1

    df2['Coder ID'] = 2
    df2['OG_Sentence ID'] = df2.index + 1

    df_concat_coder_all = pd.concat([df1, df2])
    df_concat_coder_all['Sentence ID'] = (
        df_concat_coder_all.groupby(['Sentence']).ngroup() + 1
    )

    for column in df_concat_coder_all[cal_columns]:
        k_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
            df_concat_coder_all,
            experiment_col='Sentence ID',
            annotator_col='Coder ID',
            class_col=str(column),
        )
        print('-' * 20)
        k_alpha_dict['K-alpha ' + str(column)] = k_alpha
        print(f"Krippendorff's alpha ({str(column)}): ", k_alpha)

        print('-' * 20)
    print('\n')
    for column in df_concat_coder_all[cal_columns]:
        for index, coder_number in enumerate(coders_numbers):
            coder = (
                df_concat_coder_all.loc[
                    df_concat_coder_all['Coder ID'] == coder_number, str(column)
                ]
                .astype(int)
                .to_list()
            )
            coder_score_dict[str(column)].append(
                [[int(index + 1), i, coder[i]] for i in range(len(coder))]
            )

    for column in cal_columns:
        counter = 0
        formatted_codes = coder_score_dict[str(column)][counter]
        while counter < len(coders_numbers) - 1:
            try:
                formatted_codes += coder_score_dict[str(column)][counter + 1]
                counter += 1
            except Exception:
                break

        ratingtask = agreement.AnnotationTask(data=formatted_codes)

        ir_all_dict['IR K-alpha ' + str(column)] = ratingtask.alpha()
        ir_all_dict['IR Cohen-kappa ' + str(column)] = ratingtask.kappa()
        ir_all_dict['IR Scott-pi ' + str(column)] = ratingtask.pi()

        print('-' * 20, '\n')
        print(f"Krippendorff's alpha ({str(column)}):", ratingtask.alpha())
        print(f"Cohen's Kappa ({str(column)}): ", ratingtask.kappa())
        print(f"Scott's pi ({str(column)}): ", ratingtask.pi())
        print('-' * 20, '\n')

        if save_enabled is True:
            with open(
                f'{args["parent_dir"]}{column}_FINAL_IR_all_{ir_file_name}.json', 'w', encoding='utf8') as f:
                json.dump(ir_all_dict, f)
    print('-' * 20)

    return ir_all_dict

# %%

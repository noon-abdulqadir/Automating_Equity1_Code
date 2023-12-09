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

from statannotations_fork.PValueFormat import Formatter
from statannotations_fork.stats.StatResult import StatResult


class Annotation:
    """
    Holds data, linked structs and an optional Formatter.
    """
    def __init__(self, structs, data: Union[str, StatResult],
                 formatter: Formatter = None):
        """
        :param structs: plot structures concerned by the data
        :param data: a string or StatResult to be formatted by a formatter
        :param formatter: A Formatter object. Statannotations provides a
            PValueFormatter for StatResult objects.
        """
        self.structs = structs
        self.data = data
        self.is_custom = isinstance(data, str)
        self.formatter = formatter

    @property
    def text(self):
        if self.is_custom:
            return self.data
        else:
            if self.formatter is None:
                raise ValueError('Missing a PValueFormat object to '
                                 'format the statistical result.')
            return self.formatter.format_data(self.data)

    @property
    def formatted_output(self):
        if isinstance(self.data, str):
            return self.data
        else:
            return self.data.formatted_output

    def print_labels_and_content(self, sep=' vs. '):
        labels_string = sep.join(str(struct['label'])
                                 for struct in self.structs)

        print(f'{labels_string}: {self.formatted_output}')

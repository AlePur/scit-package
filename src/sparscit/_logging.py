from collections.abc import MutableMapping
from typing import Any, Literal

_verbosity_type = Literal['all', 'no_info', 'error_only']
from tqdm import tqdm


class PBar:
    @staticmethod
    def tqdm(total: int | None = None, kwarg: dict = {}):
        kwarg = kwarg.copy()
        if total is not None:
            kwarg['total'] = total
        return tqdm(**kwarg)

class Logger:

    def __init__(self):
        self.v: _verbosity_type = 'all'

    def info(self, msg: str) -> None:
        if self.v == 'all':
            print(f"Info: {msg}")

    def warning(self, msg: str) -> None:
        if self.v in set(['all', 'no_info']):
            print(f"Warning: {msg}")

    def warn(self, msg: str) -> None:
        return self.warning(msg)

    def error(self, msg: str) -> None:
        print(f"Error: {msg}")


logging = Logger()


def set_verbosity_level(
        v: _verbosity_type
) -> None:
    """Set the global logging verbosity level.

    Parameters
    ----------
    v
        Verbosity level: ``'all'`` shows info, warnings, and errors;
        ``'no_info'`` shows warnings and errors only;
        ``'error_only'`` shows errors only
    """
    if v not in set(['all', 'no_info', 'error_only']):
        raise ValueError('invalid verbosity level')
    logging.v = v

def warn_overwrite(
        key: str,
        loc: MutableMapping[str, Any]
) -> None:
    if key in loc:
        logging.warning(f"overwriting {key}. You have probably already run this before")

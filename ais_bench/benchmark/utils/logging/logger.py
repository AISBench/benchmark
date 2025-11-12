import logging
from ais_bench.benchmark.utils.logging.error_codes import error_manager, ErrorType
from ais_bench.benchmark.global_consts import LOG_LEVEL
from ais_bench.benchmark.utils.logging.error_codes import BaseErrorCode

# custom color
class Colors:
    # content color
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    WHITE = '\033[37m'

    # background color
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_WHITE = '\033[47m'

    # style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'  # reset

LOG_COLORS = {
    logging.DEBUG: Colors.BLUE,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.BOLD + Colors.RED,
    logging.CRITICAL: Colors.BG_RED,
}

LOG_NAME = "ais_bench"
SUBPROCESS_LOG_LEVEL = logging.ERROR
LOG_NORMAL_FORMATTER = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
LOG_DEBUG_FORMATTER = '[%(asctime)s] [%(name)s] [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s'

# class to change log formatter
class ColoredLevelFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG:
            self._style._fmt = LOG_DEBUG_FORMATTER
        else:
            self._style._fmt = LOG_NORMAL_FORMATTER
        level_color = LOG_COLORS.get(record.levelno, Colors.RESET)
        record.levelname = f"{level_color}{record.levelname}{Colors.RESET}"
        return super().format(record)


def to_error_code_format(msg):
    return f"{Colors.BOLD}{Colors.BG_RED}{Colors.YELLOW}{msg}{Colors.RESET}"


def to_url_format(msg):
    return f"{Colors.UNDERLINE}{Colors.MAGENTA}{msg}{Colors.RESET}"


def to_code_msg_format(msg):
    return f"{Colors.BOLD}{Colors.RED}{msg}{Colors.RESET}"


def get_formatted_log_content(code_str, msg):
    error_code = error_manager.get(code_str)
    if not error_code:
        raise ValueError(f"error code {code_str} not found")
    formatted_msg = f"[{to_error_code_format(code_str)}]{to_code_msg_format(error_code.message)}. {msg}"

    if error_code.err_type != ErrorType.UNKNOWN:
        formatted_msg = formatted_msg + f" | Visit {to_url_format(error_code.faq_url)} for further help."

    return formatted_msg


class AISLogger:
    def __init__(self, name=LOG_NAME, level=LOG_LEVEL, is_main_process=True, log_file=None, file_mode='w'):
        """
        Args:
            name (str): Logger name.
            level (int): Log level. Default: LOG_LEVEL.
            is_main_process (bool): Whether the logger is used in main process. Default: True.
            log_file (str): Log file path. Default: None.
            file_mode (str): Log file mode. Default: 'w'.
        """
        self.formatter = ColoredLevelFormatter()
        self.logger = logging.getLogger(name)

        # clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # disable propagation to root logger to avoid duplicate logs
        self.logger.propagate = False

        for handler in self.logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        if is_main_process and log_file is not None:
            file_handler = logging.FileHandler(log_file, file_mode)
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)

        if is_main_process:
            self.logger.setLevel(level)
        else:
            self.logger.setLevel(SUBPROCESS_LOG_LEVEL)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        stream_handler.setLevel(level)
        self.logger.addHandler(stream_handler)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, stacklevel=2, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, stacklevel=2, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, stacklevel=2, *args, **kwargs)

    def error(self, error_code, msg, *args, **kwargs):
        if not isinstance(error_code, BaseErrorCode):
            raise ValueError(f"error_code {error_code} is not instance of BaseErrorCode!")
        formatted_msg = get_formatted_log_content(error_code.full_code, msg)
        self.logger.error(formatted_msg, stacklevel=2, *args, **kwargs)

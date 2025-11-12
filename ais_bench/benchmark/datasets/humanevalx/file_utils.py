# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from functools import reduce
import os
import stat

MAX_PATH_LENGTH = 4096
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_FILENUM_PER_DIR = 1024
MAX_LINENUM_PER_FILE = 10 * 1024 * 1024

FLAG_OS_MAP = {
    'r': os.O_RDONLY, 'r+': os.O_RDWR,
    'w': os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
    'w+': os.O_CREAT | os.O_TRUNC | os.O_RDWR,
    'a': os.O_CREAT | os.O_APPEND | os.O_WRONLY,
    'a+': os.O_CREAT | os.O_APPEND | os.O_RDWR,
    'x': os.O_CREAT | os.O_EXCL,
    "b": getattr(os, "O_BINARY", 0)
}


def safe_open(file_path: str, mode='r', encoding=None, permission_mode=0o600, is_exist_ok=True, **kwargs):
    """
    :param file_path: 文件路径
    :param mode: 文件打开模式
    :param encoding: 文件编码方式
    :param permission_mode: 文件权限模式
    :param is_exist_ok: 是否允许文件存在
    :param max_path_length: 文件路径最大长度
    :param max_file_size: 文件最大大小，单位: 字节, 默认值10MB
    :param check_link: 是否校验软链接
    :param kwargs:
    :return:
    """
    max_path_length = kwargs.get('max_path_length', MAX_PATH_LENGTH)
    max_file_size = kwargs.get('max_file_size', MAX_FILE_SIZE)
    check_link = kwargs.get('check_link', False)

    file_path = standardize_path(file_path, max_path_length, check_link)
    check_file_safety(file_path, mode, is_exist_ok, max_file_size)

    flags = []
    for item in list(mode):
        if item == "+" and flags:
            flags[-1] = f"{flags[-1]}+"
            continue
        flags.append(item)
    flags = [FLAG_OS_MAP.get(mode, os.O_RDONLY) for mode in flags]
    total_flag = reduce(lambda a, b: a | b, flags)

    return os.fdopen(os.open(file_path, total_flag, permission_mode),
                     mode, encoding=encoding)


def standardize_path(path: str, max_path_length=MAX_PATH_LENGTH, check_link=True):
    """
    check path
    param: path
    return: data real path after check
    """
    check_path_is_none(path)
    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)
    path = os.path.realpath(path)
    return path


def is_path_exists(path: str):
    return os.path.exists(path)


def check_path_is_none(path: str):
    if path is None:
        raise TypeError("The file path should not be None.")


def check_path_is_link(path: str):
    if os.path.islink(os.path.normpath(path)):
        raise ValueError("The path should not be a symbolic link file. "
                         f"Please check the input path:{path}.")


def check_path_length_lt(path: str, max_path_length=MAX_PATH_LENGTH):
    path_length = path.__len__()
    if path_length > max_path_length:
        raise ValueError(f"The length of path should not be greater than {max_path_length}, but got {path_length}. "
                         f"Please check the input path within the valid length range:{path[:max_path_length]}.")


def check_file_size_lt(path: str, max_file_size=MAX_FILE_SIZE):
    file_size = os.path.getsize(path)
    if file_size > max_file_size:
        raise ValueError(f"The size of file should not be greater than {max_file_size}, but got {file_size}. "
                         f"Please check the input path:{path}.")


def check_owner(path: str):
    """
    check the path owner
    param: the input path
    """
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    cur_uid = os.geteuid()
    cur_gid = os.getgid()
    if not (cur_uid == 0 or cur_uid == path_owner or path_gid == cur_gid):
        raise PermissionError(f"The current user does not have permission to access the path:{path}. "
                              "Because he is not root or the path owner, "
                              "and not in the same user group with the path owner. "
                              "Please check and make sure to satisfy at least one of the conditions above.")


def check_other_write_permission(file_path: str):
    """
    check if the specified file is writable by others who are neither the owner nor in the group
    param: the path to the file to be checked
    """
    # Get the status of the file
    file_stat = os.stat(file_path)
    # Get the mode (permission) of the file
    mode = file_stat.st_mode
    # check the write permission for others
    if mode & stat.S_IWOTH:
        raise PermissionError("The file should not be writable by others who are neither the owner nor in the group. "
                              f"Please check the input path:{file_path}, and change mode to {mode & ~stat.S_IWOTH}.")


def check_path_permission(file_path: str, is_internal_file=False):
    check_inputfiles_permission = os.getenv("MINDIE_CHECK_INPUTFILES_PERMISSION", "1") != "0"
    check_permission_flag = is_internal_file or check_inputfiles_permission
    if check_permission_flag:
        check_owner(file_path)
        check_other_write_permission(file_path)


def check_file_safety(file_path: str, mode='r', is_exist_ok=True,
                      max_file_size=MAX_FILE_SIZE, is_check_file_size=True):
    if is_path_exists(file_path):
        if not is_exist_ok:
            raise FileExistsError("The file is expected not to exist, but it already does. "
                                  f"Please check the input path:{file_path}.")
        if is_check_file_size:
            check_file_size_lt(file_path, max_file_size)
        file_dir = file_path
    else:
        if mode == 'r' or mode == 'r+':
            raise FileNotFoundError("The file is expected to exist, but it does not. "
                                    f"Please check the input path:{file_path}.")
        file_dir = os.path.dirname(file_path)

    check_path_permission(file_dir)


def safe_listdir(file_path: str, max_file_num=MAX_FILENUM_PER_DIR):
    filenames = os.listdir(file_path)
    file_num = len(filenames)
    if file_num > max_file_num:
        raise ValueError(f"The file num in dir is {file_num}, which exceeds the limit {max_file_num}. "
                         f"Please check the input path:{file_path}.")
    return filenames


def safe_chmod(file_path: str, permission_mode):
    standard_path = standardize_path(file_path)
    check_path_permission(standard_path)
    os.chmod(file_path, permission_mode)


def has_owner_write_permission(file_path: str):
    st = os.stat(file_path)
    return st.st_mode & stat.S_IWUSR


def safe_readlines(file_obj, max_line_num=MAX_LINENUM_PER_FILE):
    lines = file_obj.readlines()
    line_num = len(lines)
    if line_num > max_line_num:
        raise ValueError(f"The file line num is {line_num}, which exceeds the limit {max_line_num}. "
                         f"Please check the input file:{file_obj.name}.")
    return lines

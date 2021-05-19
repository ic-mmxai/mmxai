import os
import sqlite3
import random
from datetime import datetime
"""
This library includes functions for uploaded file management and database update
Functions will be called by APScheduler in app.py
Confiuration can be found in config.py
"""


def generate_random_str(random_length=16):
    """
    This method will generate a random bits(dafult to be 16) long string, which is used as the name of new user's directory
    :param random_length: desired length of string
    :return: randomly generated string
    """
    random_str = ""
    base_str = "ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789"
    length = len(base_str) - 1
    for i in range(random_length):
        random_str += base_str[random.randint(0, length)]
    return random_str


def mkdir(path):
    """
    This function can create a new directory
    :param path: file path
    :return: boolean:hether the directory has been created
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def clean(path):
    """
    This method will delete all files under "path"
    :param path: file path
    :return:
    """
    clear_dir(path)
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        os.rmdir(path_file)


def clear_dir(path):
    """
    This method will delete all files and directories under "path", then delete the parent directory.
    :param path: file path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        return
    else:
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)


def check_database(object):
    """
    Periodic method called by APScheduler in app.py.
    It can check all user information in our local database and delete all expired information.
    Additionally, check the "staitic/user" directory to erase expired directories as well.
    """
    db = 'user_info.sqlite3'
    con = sqlite3.connect(db)
    cur = con.cursor()
    select_sql = "select file_name,expired_time,id,ip_addr from user_info"
    cur.execute(select_sql)
    date_set = cur.fetchall()
    cur.close()
    cur = con.cursor()
    for data in date_set:
        if datetime.now() > datetime.strptime(data[1], "%Y-%m-%d %H:%M:%S.%f"):
            delete_sql = "delete from user_info where id = %d"%int(data[2])
            cur.execute(delete_sql)
    con.commit()
    cur.close()
    cur = con.cursor()
    select_sql = "select file_name,expired_time,id,ip_addr from user_info"
    cur.execute(select_sql)
    date_set = cur.fetchall()
    cur.close()
    for dir in os.listdir("./static/user"):
        flag = 0
        for row in date_set:
            if dir == row[0] and datetime.now() < datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f"):
                flag = 1
        if flag == 0:
            file_path = './static/user/' + dir
            clear_dir(file_path)
            os.rmdir(file_path)
    con.close()

# -- coding: utf-8 --

import sys
import getopt
import os

bin_file_path = None
model_sign = None
try:
    opts, args = getopt.getopt(sys.argv[1:], '-h-p:-s:', ['path=', 'sign='])
except getopt.GetoptError:
    print('Usage: .bin_filename_modification.py -p <path> -s <sign>')
    sys.exit(2)

# 如果无opts或有args, 报错
if not opts or args:
    print('Usage: .bin_filename_modification.py -p <path> -s <sign>')
    sys.exit(2)
else:
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: .bin_filename_modification.py -p <path> -s <sign>')
            sys.exit()
        elif opt in ('-p', '--path'):
            bin_file_path = arg
        elif opt in ('-s', '--sign'):
            model_sign = arg


# 获取该目录下所有文件, 存入列表中
file_list = os.listdir(bin_file_path)

for file in file_list:
    try:
        int(file.split('.')[-2].split('_')[-1])  # 过滤掉已经被标记的bin文件
        if file.split('.')[-1] == 'bin' and file.split('.')[-2].split('_')[-1] != model_sign:
            old_name = bin_file_path + file
            new_name = bin_file_path + file.split('.')[0] + '_' + model_sign + '.bin'
            os.rename(old_name, new_name)
    except:
        pass


# -- coding: utf-8 --

import getopt
import os
import re
import sys

h_file_name = None
model_sign = None

# 设置变量名称file sign
try:
    opts, args = getopt.getopt(sys.argv[1:], '-h-f:-s:', ['file=', 'sign='])
except getopt.GetoptError:
    print('Usage: .h_varname_modification.py -f <file> -s <sign>')
    sys.exit(2)

# 如果无opts或有args, 报错
if not opts or args:
    print('Usage: .h_varname_modification.py -f <file> -s <sign>')
    sys.exit(2)
else:
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: .h_varname_modification.py -f <file> -s <sign>')
            sys.exit()
        elif opt in ('-f', '--file'):
            h_file_name = arg
        elif opt in ('-s', '--sign'):
            model_sign = arg


# 获取txt总行数
total_rows = len(open(h_file_name).readlines())

# 读取文件所有行
lines = []
f = open(h_file_name, 'r')  # file path
for line in f:
    lines.append(line)
f.close()

# 找出要替换的变量名
name_list = []
for line in lines:
    if line.split(' ')[0] == '#define':
        variable_name = line.split(' ')[1]
        name_list.append(variable_name)
    if line.split(' ')[0] == 'const':
        variable_name = line.split(' ')[2].strip('[]')
        name_list.append(variable_name)
name_list.append('nnom_input_data')
name_list.append('nnom_output_data')
name_list.append('nnom_model_create')

# 全文搜索并替换 添加后缀
lines = ''.join(lines)
for i in range(len(name_list)):
    lines = re.sub('\\b' + name_list[i] + '\\b', name_list[i] + '_' + model_sign, lines)

# 重新写入文件
f = open(h_file_name, 'w+')
f.write(lines)
f.close()

# 文件重命名
os.rename(h_file_name, '.'.join(h_file_name.split('.')[:-1]) + '_' + model_sign + '.' + h_file_name.split('.')[-1])

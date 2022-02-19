# -- coding: utf-8 --

import sys

# 文件路径
try:
    cc_file_name = sys.argv[1]
except:
    cc_file_name = './1.cc'

total_rows = len(open(cc_file_name).readlines())  # 获取txt总行数

# 读取文件所有行
lines = []
f = open(cc_file_name, 'r')  # file path
for line in f:
    lines.append(line)
f.close()

# 检测是否已经改过
if lines[0].strip().split()[0] == 'unsigned':
    # 更改头行
    head_row_split = lines[0].strip().split()
    head_row_split[2] = 'model_tflite[]'
    head_row_split.insert(0, '#include "model.h"\n\n\n'
                             '// Name of model tflite flatbuffer.\n'
                             'const unsigned char model_tflite_name[] = {"model"};\n\n'
                             '// Model data tflite flatbuffer.\n'
                             'const')
    head_row = ' '.join(head_row_split)

    # 更改尾行
    tail_row_split = lines[total_rows - 1].strip().split()
    tail_row_split[2] = 'model_tflite_len'
    tail_row_split.insert(0, 'const')
    tail_row = ' '.join(tail_row_split)

    # 将更改后的行加入到所有行内
    lines[0] = head_row
    lines.insert(1, '\n')
    lines[total_rows] = tail_row
    lines.insert(total_rows + 1, '\n')

    # 重新写入文件
    f = open(cc_file_name, 'w+')
    f.write(''.join(lines))
    f.close()

    print('.cc_file_modification.py: manually reformat successfully!')
else:
    print('.cc_file_modification.py: The .cc file does not need to be reformatted!')

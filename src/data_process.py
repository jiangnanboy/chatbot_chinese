import os
import re

def process_raw_chatbot_data():
    '''
    将原始数据分为source与target两个文件（或者写在一个文件中，一行存储source与target，中间用空格分隔）
    :return:
    '''
    #存储source数据
    chat_source_path = os.path.join(os.getcwd(), "chat_source.txt")
    #存储target数据
    chat_target_path = os.path.join(os.getcwd(), "chat_target.txt")
    #原始数据
    chat_data_path = os.path.join(os.getcwd(), "chat_data.txt")
    with open(chat_data_path, 'r', encoding='utf-8') as raw_f:
        with open(chat_source_path, 'w', encoding='utf-8') as source_f, open(chat_target_path, 'w', encoding='utf-8') as target_f:
            source_target_flag = 0
            for line in raw_f:
                line = line.strip()
                if line == 'E':
                    continue
                elif source_target_flag == 0:
                    line = line[2:len(line)]
                    words=''
                    for word in line:
                        if word !=' ':
                            words+=word
                    line = ' '.join(list(words))
                    source_f.write(line)
                    source_f.write('\n')
                    source_target_flag = 1
                elif source_target_flag == 1:
                    line = line[2:len(line)]
                    words=''
                    for word in line:
                        if word !=' ':
                            words+=word
                    line = ' '.join(list(words))
                    target_f.write(line)
                    target_f.write('\n')
                    source_target_flag = 0
    print('source and target data process done!')

def regular(line):
    '''
    规范化句子
    '''
    line = line.replace('/', '')
    line = re.sub(r'…{1,100}', '…', line)
    line = re.sub(r'\.{3,100}', '…', line)
    line = re.sub(r'···{2,100}', '…', line)
    line = re.sub(r',{1,100}', '，', line)
    line = re.sub(r'\.{1,100}', '。', line)
    line = re.sub(r'。{1,100}', '。', line)
    line = re.sub(r'\?{1,100}', '？', line)
    line = re.sub(r'？{1,100}', '？', line)
    line = re.sub(r'!{1,100}', '！', line)
    line = re.sub(r'！{1,100}', '！', line)
    line = re.sub(r'~{1,100}', '～', line)
    line = re.sub(r'～{1,100}', '～', line)
    line = re.sub(r'[“”]{1,100}', '"', line)
    line = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', line)
    line = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', line)
    return line

def filter_source_target_chatbot_data():
    '''
    这里是将经过process_raw_chatbot_data()处理过的两个文件中的数据进行规范化并合并成一个文本中
    :return:
    '''
    chat_source_target_path = os.path.join(os.getcwd(), "chat_source_target.txt")
    chat_source_path = os.path.join(os.getcwd(), "chat_source.txt")
    chat_target_path = os.path.join(os.getcwd(), "chat_target.txt")
    with open(chat_source_target_path, 'w', encoding="utf-8") as source_target_f, open(chat_source_path, 'r', encoding="utf-8") as source_f, open(chat_target_path, 'r', encoding="utf-8") as target_f:
        for source_line, target_line in zip(source_f,target_f):
            source_line = regular(source_line)
            target_line = regular(target_line)
            if (len(source_line.strip()) == 0) or (len(target_line.strip()) == 0):
                continue
            source_line = ' '.join(source_line)
            target_line = ' '.join(target_line)
            source_target_f.write(source_line + "@@@@" + target_line)
            source_target_f.write("\n")
    print('source and target data write to file done!')

def process_source_target_data():
    '''
    这里将经过filter_source_target_chatbot_data()处理的数据再次分成两个文件。
    :return:
    '''
    chat_source_target_path = os.path.join(os.getcwd(), 'chat_source_target.txt')
    #存储source数据
    chat_source_path = os.path.join(os.getcwd(), "chat_source.src")
    #存储target数据
    chat_target_path = os.path.join(os.getcwd(), "chat_target.trg")
    with open(chat_source_target_path, 'r', encoding='utf-8') as raw_f,open(chat_source_path, 'w', encoding="utf-8") as source_f, open(chat_target_path, 'w', encoding="utf-8") as target_f:
        for line in raw_f:
            line = line.split('@@@@')
            source_f.write(line[0])
            source_f.write('\n')
            target_f.write(line[1])
    print('process source and target data done!')
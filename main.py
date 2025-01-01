# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv

train_test_ratio=0.8
def getData():
    falsenews = {
        'title': [],
        'label': []
    }
    truenews = {
        'title': [],
        'label': []
    }
    train_data={
        'title': [],
        'label': []
    }
    test_data={
        'title': [],
        'label': []
    }
    #获取所有的新闻标题和标签，并将标签转为0和1，还要实现去除没有新闻标题的选项
    with open('./data/news.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        list=[]
        for row in reader:
            list.append(row)
        for i in range(1,len(list)):
            if list[i][6]!='' and list[i][4]=='谣言':
                falsenews['title'].append(list[i][6])
                falsenews['label'].append(0)
            elif list[i][6]!='' and list[i][4]=='事实':
                truenews['title'].append(list[i][6])
                truenews['label'].append(1)
    # print(len(falsenews['title']))
    # print(len(truenews['title']))
    #将数据划分为训练集和测试集，分别写入test_data.csv和train_data.csv文件中，划分比例为全局变量train_test_ratio
    for index1 in range(0,math.floor(len(falsenews['title'])*train_test_ratio)):
        train_data['title'].append(falsenews['title'][index1])
        train_data['label'].append((falsenews['label'][index1]))
    for index2 in range(0,math.floor(len(truenews['title'])*train_test_ratio)):
        train_data['title'].append(truenews['title'][index2])
        train_data['label'].append((truenews['label'][index2]))
    for index1 in range(math.floor(len(falsenews['title'])*train_test_ratio),len(falsenews['title'])):
        test_data['title'].append(falsenews['title'][index1])
        test_data['label'].append((falsenews['label'][index1]))
    for index2 in range(math.floor(len(truenews['title'])*train_test_ratio),len(truenews['title'])):
        test_data['title'].append(truenews['title'][index2])
        test_data['label'].append((truenews['label'][index2]))
    with open('./data/train_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        for title, label in zip(train_data['title'], train_data['label']):
            writer.writerow([title, label])  # 将每一对 title 和 label 写入一行
    with open('./data/test_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        for title, label in zip(test_data['title'], test_data['label']):
            writer.writerow([title, label])  # 将每一对 title 和 label 写入一行


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# def getTF_IDFVector:
#     df=pd.DataFrame()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    getData()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

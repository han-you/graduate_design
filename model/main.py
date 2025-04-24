# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import jieba
import csv
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from keras.api.models import load_model
import joblib

# from tensorflow.python.keras.saving.save import load_model

max_features=1000
train_test_ratio=0.8
zero_rate=1.4
one_rate=0.8
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

    # conn = pymysql.connect(
    #     host="123.57.251.203",
    #     user='hanyou',
    #     password='Chenyu&20021122',
    #     database='news',
    #     port=3306
    # )
    # cursor = conn.cursor()
    # sql = 'select * from news'
    # cursor.execute(sql)
    # data = cursor.fetchall()
    # cursor.close()
    # conn.close()
    #
    # with open('data/news.csv', mode='a', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     for row in data:
    #         writer.writerow(['','','','','事实','',row[0]])
    #
    # for row in data:
    #     truenews['title'].append(row[0])
    #     truenews['label'].append('事实')

    # 获取所有的新闻标题和标签，并将标签转为0和1，还要实现去除没有新闻标题的选项
    with open('data/news.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        list = []
        for row in reader:
            list.append(row)
        # print(len(list))
        for i in range(1, len(list)):
            if list[i][6] != '' and list[i][4] == '谣言':
                falsenews['title'].append(list[i][6])
                falsenews['label'].append(0)
            elif list[i][6] != '' and list[i][4] == '事实':
                truenews['title'].append(list[i][6])
                truenews['label'].append(1)
    # print(len(falsenews['title']))
    # print(len(truenews['title']))
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
    with open('data/train_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        for title, label in zip(train_data['title'], train_data['label']):
            writer.writerow([title, label])  # 将每一对 title 和 label 写入一行
    with open('data/test_data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        for title, label in zip(test_data['title'], test_data['label']):
            writer.writerow([title, label])  # 将每一对 title 和 label 写入一行

# 假设停用词表保存在 stopwords.txt 中
def remove_stopwords(words):
    with open('data/stop_words.txt', 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    return [word for word in words if word not in stopwords]

#进行分词
def divideWords(dataset_name):
    titles=[]
    with open(dataset_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        list=[]
        for row in reader:
            list.append(row)
        # print(list[0])
        for i in range(0, len(list)):
            titles.append(list[i][0])
    # 对每个标题进行分词
    ans=[]
    for title in titles:
        words = jieba.cut(title)
        filtered_words = remove_stopwords(words)
        ans.append(" ".join(filtered_words))  # 输出去除停用词后的分词结果
    return ans



#得到标签
def getlabel(dataset_name):
    labels=[]
    with open(dataset_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        list = []
        for row in reader:
            list.append(row)
        for i in range(0, len(list)):
            if list[i][1]=='1':
                labels.append(1)
            else:
                labels.append(0)
    return labels

# 使用 TfidfVectorizer 进行向量化
def toVector_train(wordvectorlist):
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_tfidf = vectorizer.fit_transform(wordvectorlist).toarray()
    joblib.dump(vectorizer, '../myapp/vectorizer.pkl')  # 保存vectorizer
    return x_tfidf

def toVector_test(wordvectorlist):
    # print(wordvectorlist[0:5])
    vectorizer = joblib.load('../myapp/vectorizer.pkl')  # 加载已保存的vectorizer
    x_tfidf = vectorizer.transform(wordvectorlist).toarray()  # 只转换
    return x_tfidf

def DNN(x_train,y_train,x_test,y_test):
    # 创建神经网络模型
    model = Sequential()

    # 输入层：假设 X 的每个样本有 1000 个特征
    model.add(Dense(512, input_shape=(max_features,), activation='relu'))  # 第一个隐藏层（512个神经元）
    model.add(Dropout(0.5))  # Dropout层，避免过拟合

    model.add(Dense(256, activation='relu'))  # 第二个隐藏层（256个神经元）
    model.add(Dropout(0.5))  # Dropout层

    model.add(Dense(128, activation='relu'))  # 第三个隐藏层（128个神经元）
    model.add(Dropout(0.5))  # Dropout层

    model.add(Dense(1, activation='sigmoid'))  # 输出层（sigmoid激活函数用于二分类）

    # 编译模型 
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # 打印模型架构
    # model.summary()

    # 训练模型
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # 设置早停
    # print(x_train[0])
    # print(x_train[1])
    # print(x_train.dtype)  # 检查 x_train 的数据类型
    # print(y_train.dtype)  # 检查 y_train 的数据类型
    # print(y_train)
    # 假设 y_train 是标签
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    history = model.fit(x_train, y_train,
                        shuffle=True,
                        epochs=20,  # 可以根据需要调整
                        batch_size=32,  # 可以根据需要调整
                        validation_data=(x_test, y_test),  # 验证集
                        # class_weight={0:class_weights[0]*zero_rate,1:class_weights[1]*one_rate},
                        callbacks=[early_stopping])  # 使用早停法

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    # predictions = model.predict(x_test[2810:2821])
    # print(predictions)
    # print(np.count_nonzero(predictions > 0.5) / len(predictions))
    # with open("results.txt", 'a') as file:
    #     file.write(f"zero rate:{zero_rate}; "+f"one rate:{one_rate}; "+f"loss:{loss}; "+f"accuracy:{accuracy}; "+f"predict:{np.count_nonzero(predictions > 0.5) / len(predictions)}\n")
    model.save('my_model.h5')

if __name__ == '__main__':
    # getData()

    x_train=toVector_train(divideWords('./data/train_data.csv'))
    # count=0
    # for row in x_train:
    #     for num in row:
    #         if num!=0:
    #             count+=1
    # print(count)
    y_train=getlabel('./data/train_data.csv')
    x_test=toVector_test(divideWords('./data/test_data.csv'))
    y_test=getlabel('./data/test_data.csv')
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    # x_train=toVector(x_train)
    # x_test=toVector(y_test)
    # for i in np.arange(2.0,4.0,0.1):
    #     zero_rate=i
    #     one_rate=0.8
    #     DNN(x_train, y_train, x_test, y_test)
    DNN(x_train, y_train, x_test, y_test)
    # model=load_model('my_model.h5')
    # predictions=model.predict(x_test)
    # print(np.count_nonzero(predictions > 0.28505) / len(predictions))
    # for item in vector:
    #     print(str(len(item))+' ')
    # print(len(wordvectorlist) == len(labellist))
    # print(wordvectorlist)
    # print(labellist)


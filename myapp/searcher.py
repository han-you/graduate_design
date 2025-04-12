import csv
import math
import operator
import time

import jieba
import joblib
import numpy
from keras.api.models import load_model
from tensorflow.python.autograph.operators import list_append

from myapp.indexer import Indexer
from sklearn.feature_extraction.text import TfidfVectorizer


class Search:

    def __init__(self,k1=1.5,b=0.5,coefficient=0.5):

        self.coefficient=coefficient
        self.model=load_model('my_model.h5')
        self.index=Indexer()
        self.k1 = k1
        self.b = b
        self.avgdl = self.avg_doc_length()  # 计算平均文档长度
        self.loaded_vectorizer=joblib.load('vectorizer.pkl')
    # 假设停用词表保存在 stopwords.txt 中
    def remove_stopwords(self,words):
        with open('stop_words.txt', 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())
        return [word for word in words if word not in stopwords]

    # 进行分词
    def divideWords(self,titles):
        ans=[]
        for title in titles:
            words = jieba.cut(title)
            # print(words)
            filtered_words = self.remove_stopwords(words)
            ans.append(" ".join(filtered_words))  # 输出去除停用词后的分词结果
        return ans

    # 使用 TfidfVectorizer 进行向量化
    def toVector_test(self,wordvectorlist):
        # print(wordvectorlist)
        x_tfidf = self.loaded_vectorizer.transform(wordvectorlist).toarray()  # 只转换
        return x_tfidf

    def search(self,query):  #BM25
        start_time = time.time()
        queries = query.split()
        term_list = []
        for item in queries:
            term_list.extend(jieba.cut_for_search(item))

        BM25 = {}
        for doc in self.index.doc_list:
            BM25[doc['title']] = self.compute_bm25_score(term_list, doc)

        # 过滤零分文档并按BM25分数排序
        sorted_doc = sorted(
            [(title, score) for title, score in BM25.items() if score != 0],
            key=lambda x: x[1],
            reverse=True
        )[:200]  # 直接取前200个结果
        print(sorted_doc)
        res = [self.index.id_doc[doc_id] for doc_id, score in sorted_doc]

        end_time = time.time()
        print(end_time-start_time)
        print('search finish')
        return res

    def avg_doc_length(self):
        total_length=sum(len(list(jieba.cut_for_search(doc['title']))) for doc in self.index.doc_list)
        return total_length/len(self.index.doc_list) if len(self.index.doc_list)!=0 else 0

    def compute_bm25_score(self,queries,doc):
        score=0.0
        doc_title=list(jieba.cut_for_search(doc['title']))
        doc_len=len(doc_title)
        for term in queries:
            if term in self.index.idf:
                f=doc_title.count(term)
                idf_value = self.index.idf[term]
                numerator=f*(self.k1+1)
                denominator=f+self.k1*(1-self.b+self.b*doc_len/self.avgdl)
                score+=idf_value*numerator/denominator
        return score

    def search2(self,query):  #BM25
        start_time=time.time()
        queries = query.split()
        term_list=[]
        for item in queries:
            term_list.extend(jieba.cut_for_search(item))
        BM25={}
        # print(term_list)
        for doc in self.index.doc_list:
            BM25[doc['title']]=self.compute_bm25_score(term_list,doc)

        tmp= {title: score for title, score in BM25.items() if score != 0}         #将无关的新闻全部删除
        tmp = dict(sorted(tmp.items(), key=lambda item: item[1], reverse=True)[:200])
        BM25=tmp


        temp = list(zip(*BM25.items()))

        if(len(temp)==0):
            return []

        list_title, list_score = list(temp[0]), list(temp[1])
        # print(list_score)
        scores = self.model.predict(numpy.array(self.toVector_test(self.divideWords(list_title))))
        # print(scores)
        min_score = min(list_score)
        max_score = max(list_score)
        if max_score > min_score:  # 避免除以零
            for i in range(0, len(list_score)):
                 list_score[i]= (list_score[i] - min_score) / (max_score - min_score)
        else:
            for i in range(0, len(list_score)):  # 所有分数相同的情况，直接设为 1
                list_score[i]=1.0

        for i in range(0, len(list_score)):
            # print(scores[i][0]*self.coefficient+(1-self.coefficient)*list_score[i])
            BM25[list_title[i]]=scores[i][0]*self.coefficient+(1-self.coefficient)*list_score[i]


        sorted_doc = sorted(BM25.items(), key=operator.itemgetter(1), reverse=True)

        res=[self.index.id_doc[doc_id] for doc_id,score in sorted_doc if score!=0]
        # for item in res:
        #     print(BM25[item['title']])
        end_time=time.time()
        print(end_time-start_time)
        print('search finish')
        return res

    def search3(self, query):  # BM25
        start_time = time.time()
        queries = query.split()
        term_list = []
        for item in queries:
            term_list.extend(jieba.cut_for_search(item))
        BM25 = {}

        # 计算原始BM25分数
        for doc in self.index.doc_list:
            BM25[doc['title']] = self.compute_bm25_score(term_list, doc)

        # 过滤并排序
        tmp = {title: score for title, score in BM25.items() if score != 0}
        tmp = dict(sorted(tmp.items(), key=lambda item: item[1], reverse=True)[:200])
        BM25 = tmp

        temp = list(zip(*BM25.items()))
        if (len(temp) == 0):
            return []

        list_title, list_score = list(temp[0]), list(temp[1])

        # 模型预测
        scores = self.model.predict(numpy.array(self.toVector_test(self.divideWords(list_title))))

        # 归一化处理
        min_score = min(list_score)
        max_score = max(list_score)
        if max_score > min_score:
            list_score = [(s - min_score) / (max_score - min_score) for s in list_score]
        else:
            list_score = [1.0 for _ in list_score]

        # 计算最终分数
        final_scores = {}
        for i in range(len(list_score)):
            title = list_title[i]
            bm25_score = list_score[i]
            model_score = scores[i][0]
            final_score = model_score * self.coefficient + (1 - self.coefficient) * bm25_score
            final_scores[title] = {
                'final_score': final_score,
                'bm25_score': bm25_score,
                'model_score': model_score
            }
            BM25[title] = final_score

        # 打印详细分数信息
        print("\n=== 搜索结果分数详情 ===")
        print(f"系数设置: 模型权重={self.coefficient}, BM25权重={1 - self.coefficient}")
        print(f"{'排名':<5}{'标题':<50}{'最终分数':<15}{'BM25分数':<15}{'模型分数':<15}")

        sorted_doc = sorted(BM25.items(), key=operator.itemgetter(1), reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_doc, 1):
            detail = final_scores[doc_id]
            print(
                f"{rank:<5}{doc_id[:50]:<50}{score:<15.4f}{detail['bm25_score']:<15.4f}{detail['model_score']:<15.4f}")

        # 返回结果
        res = [self.index.id_doc[doc_id] for doc_id, score in sorted_doc if score != 0]
        end_time = time.time()
        print(f"\n总耗时: {end_time - start_time:.4f}秒")
        print('search finish')
        return res



        # scores = list(BM25.values())
        # min_score = min(scores)
        # max_score = max(scores)
        # if max_score > min_score:  # 避免除以零
        #     for title in BM25:
        #         score=self.model.predict(numpy.array(self.toVector_test(self.divideWords(title))))
        #         print(score)
        #         BM25[title] = (BM25[title] - min_score) / (max_score - min_score)
        # else:
        #     for title in BM25:  # 所有分数相同的情况，直接设为 1
        #         BM25[title] = 1.0
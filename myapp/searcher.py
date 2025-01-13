import csv
import math
import operator
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
            print(words)
            filtered_words = self.remove_stopwords(words)
            ans.append(" ".join(filtered_words))  # 输出去除停用词后的分词结果
        return ans

    # 使用 TfidfVectorizer 进行向量化
    def toVector_test(self,wordvectorlist):
        # print(wordvectorlist)
        x_tfidf = self.loaded_vectorizer.transform(wordvectorlist).toarray()  # 只转换
        return x_tfidf

    def search(self,query):  #TF-IDF
        term_list=[]
        query=query.split()
        for item in query:
            term_list.extend(jieba.cut_for_search(item))
        # print(term_list)
        tf_idf={}
        for item in term_list:
            # print(self.index.inverted[item])
            if item in self.index.inverted:
                for doc_id,fre in self.index.inverted[item].items():
                    print(doc_id)
                    if doc_id in tf_idf:
                        tf_idf[doc_id]+=(1+math.log10(fre))*self.index.idf[item]
                    else:
                        tf_idf[doc_id]=(1+math.log10(fre))*self.index.idf[item]

        sorted_doc=sorted(tf_idf.items(),key=operator.itemgetter(1),reverse=True)
        res=[self.index.id_doc[doc_id] for doc_id,score in sorted_doc]
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
        queries = query.split()
        BM25={}

        for doc in self.index.doc_list:
            BM25[doc['title']]=self.compute_bm25_score(queries,doc)

        tmp= {title: score for title, score in BM25.items() if score != 0}
        BM25=tmp                           #将无关的新闻全部删除

        temp = list(zip(*BM25.items()))
        list_title, list_score = list(temp[0]), list(temp[1])
        print(list_score)
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
            print(scores[i][0]*self.coefficient+(1-self.coefficient)*list_score[i])
            BM25[list_title[i]]=scores[i][0]*self.coefficient+(1-self.coefficient)*list_score[i]


        sorted_doc = sorted(BM25.items(), key=operator.itemgetter(1), reverse=True)

        res=[self.index.id_doc[doc_id] for doc_id,score in sorted_doc if score!=0]
        # for item in res:
        #     print(BM25[item['title']])
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
import math
import operator
from itertools import count

import jieba

from index import Indexer
class Search:
    def __init__(self,k1=1.5,b=0.5):

        self.index=Indexer()
        self.k1 = k1
        self.b = b
        self.avgdl = self.avg_doc_length()  # 计算平均文档长度

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
        sorted_doc = sorted(BM25.items(), key=operator.itemgetter(1), reverse=True)
        res=[self.index.id_doc[doc_id] for doc_id,score in sorted_doc if score!=0]
        # for item in res:
        #     print(BM25[item['title']])
        print('search finish')
        return res
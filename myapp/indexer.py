import math

import jieba
import pymysql


class Indexer:
    inverted={}  #词所在文档及词频
    idf={}       #词的逆文档频率
    id_doc={}    #文档与词的对应关系

    def __init__(self):
        self.doc_num=0
        self.doc_list=[]
        self.index_writer()

    def index_writer(self):
        conn = pymysql.connect(
            host="123.57.251.203",
            user='hanyou',
            password='Chenyu&20021122',
            database='news',
            port=3306
        )
        cursor = conn.cursor()
        sql='select * from news'
        cursor.execute(sql)
        data=cursor.fetchall()
        cursor.close()
        conn.close()
        # print(data)
        for row in data:
            doc={}
            doc.setdefault('title',row[0])
            doc.setdefault('url',row[1])
            self.doc_list.append(doc)
        self.index()

    def index(self):
        self.doc_num=len(self.doc_list)
        for doc in self.doc_list:
            key=doc['title']
            #正排
            self.id_doc[key]=doc

            #倒排
            iterm_list=list(jieba.cut_for_search(key))
            for item in iterm_list:
                if item in self.inverted:
                    if key not in self.inverted[item]:
                        self.inverted[item][key]=1
                    else:
                        self.inverted[item][key]+=1
                else:
                    self.inverted[item]={key:1}

        for item in self.inverted:
            self.idf[item]=math.log10(self.doc_num/len(self.inverted[item]))

        print("inverted terms:%d" % len(self.inverted))
        print("index done")




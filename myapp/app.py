import jieba
from flask import Flask, render_template, request
from myapp.searcher import Search
app = Flask(__name__)

#创建Search类实例
search_engine=Search()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    # 使用 Search 类的 search2 方法进行 BM25 搜索
    search_results = search_engine.search2(query)  # 或者使用 search(query) 进行 TF-IDF 搜索
    tmp=highlight(search_results,list(jieba.cut_for_search(query)))
    return render_template('results.html', query=query, docs=tmp,length=len(search_results))
# @app.route('/search/<query>', methods=['POST'])
# def search(query):
#     query = request.form['query']
#     # 根据 query 搜索新闻，可以优化为模糊匹配或者通过索引查找
#     filtered_news = [news for news in news_data if query.lower() in news['title'].lower()]
#     return render_template('results.html', query=query, news=filtered_news)

def highlight(docs, terms):
    result = []
    print()
    for doc in docs:
        content = doc['title']
        for term in terms:
            content = content.replace(term, '<em><font color="red">{}</font></em>'.format(term))
            # print(content)
        result.append((doc['url'], content))
    print(result)
    return result

if __name__ == '__main__':
    app.run(debug=True)

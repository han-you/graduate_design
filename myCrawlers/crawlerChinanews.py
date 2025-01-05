import csv
import os
import re
import time
import urllib.request,urllib.error

from bs4 import BeautifulSoup

totalpage=10

def delete_file():
    file_path='data/newsChinanews.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除")
    else:
        print(f"文件 {file_path} 不存在")

def saveData(urls,texts):
    print(urls)
    print("save...")
    with open('data/newsChinanews.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        for url, text in zip(urls, texts):
            writer.writerow([url, text])  # 将每一对 title 和 label 写入一行

def sleep(int):
    time.sleep(int)

# 得到指定 一个URL的网页内容
def askURL(url):
    headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                }
    request = urllib.request.Request(url=url,headers=headers)
    html=""
    try:
        response = urllib.request.urlopen(request)
        html=response.read().decode("utf-8","ignore")
    except urllib.error.URLError as e:
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html

def getData(baseurl):
    urls = []
    texts = []
    # url=baseurl+'page_1.html'
    # html=askURL(url)
    # soap=BeautifulSoup(html,'html.parser')
    # tmp=soap.find_all('a',attrs={'shape':'rect'})
    # print(tmp)
    for i in range(1,1+totalpage):
        print(i)
        followurl='news'+str(i)+'.html'
        url=baseurl+followurl
        html=askURL(url)#获取网页源码
        #逐一解析数据
        soap=BeautifulSoup(html,'html.parser')
        for item in soap.find_all('div',attrs={'class':'dd_bt'}):
            item=str(item)
            pattern = re.compile(r'<a href="(.*?)".*?>(.*?)</a>', re.S)
            URL,text = pattern.findall(item)[0]
            # print(result)
            print([URL, text])
            # break
            urls.append('https://'+'www.chinanews.com.cn' + URL)
            texts.append(text)
        sleep(3)
    return urls,texts
def main():
    delete_file()
    baseurl="https://www.chinanews.com.cn/scroll-news/"
    # 1.爬取网页
    [urls,texts] = getData(baseurl)
    saveData(urls,texts)


if __name__=='__main__':
    main()
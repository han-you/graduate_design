import mysql.connector
from selenium import webdriver
import csv
import os
import re
import time
import urllib.request,urllib.error
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

totalpage=45


def delete_file():
    file_path='newsSina.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除")
    else:
        print(f"文件 {file_path} 不存在")

def saveData(urls,texts):
    print(urls)
    print(texts)
    print("save...")
    with open('newsSina.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        for url, text in zip(urls, texts):
            writer.writerow([url, text])  # 将每一对 title 和 label 写入一行

def saveData2DB(urls,texts):
    conn=mysql.connector.connect(
        host="123.57.251.203",
        user='hanyou',
        password='Chenyu&20021122',
        database='news'
    )
    cursor = conn.cursor()
    for url,title in zip(urls,texts):
        cursor.execute('insert into news (title,url) values (%s,%s)',(title,url))

    conn.commit()

    cursor.close()
    conn.close()

def sleep(int):
    time.sleep(int)


def getData(baseurl,driver):
    urls = []
    texts = []
    for i in range(1,1+totalpage):
        followurl=baseurl+'page='+str(i)
        driver.get(followurl)
        driver.refresh()
        div=driver.find_element(By.ID,'d_list')
        lis=div.find_elements(By.XPATH,'.//ul//li')
        for li in lis:
            span=li.find_element(By.CLASS_NAME,'c_tit')
            a=span.find_element(By.XPATH,'.//a')
            urls.append(a.get_attribute('href'))
            texts.append(a.text)
            print(a.get_attribute('href'))
            print(a.text)
        sleep(3)
    return urls,texts

def main(driver):
    delete_file()
    baseurl="https://news.sina.com.cn/roll/#pageid=153&lid=2509&k=&num=50&"
    # # 1.爬取网页
    [urls,texts] = getData(baseurl,driver)
    saveData2DB(urls,texts)
    saveData(urls,texts)

if __name__=='__main__':
    web_driver=webdriver.Edge()
    main(web_driver)
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:19:50 2024

@author: Debabrata Ghorai, Ph.D.

Possible solution of web-scrapping.

1) Web scraping is the process of using bots to extract content and data from a website.
2) Web scraping is an automatic method to obtain large amounts of data from websites.
3) Web scraping (or data scraping) is a technique used to collect content and data from the internet. 
4) There are many different ways to perform web scraping to obtain data from websites.
5) These include using online services, particular API's or even creating your code for web scraping from scratch.
6) Web Scrapers can extract all the data on particular sites or the specific data that a user wants.

All web scraping bots follow three basic principles:
Step 1: Making an HTTP request to a server
Step 2: Extracting and parsing (or breaking down) the website's code
Step 3: Saving the relevant data locally

How to scrape the web (step-by-step):
Step 1: Find the URLs you want to scrape
Step 2: Inspect the page
Step 3: Identify the data you want to extract
Step 4: Write the necessary code
Step 5: Execute the code
Step 6: Storing the data

What tools can you use to scrape the web:
- BeautifulSoup
- Scrapy
- Pandas
- Parsehub
- APIs
- selenium, etc.
"""


import os
import time
import re
import ctypes
import urllib.request
import requests
import pandas as pd

from selenium import webdriver
# from selenium.webdriver.chrome.options import Options as ch_ops
# from selenium.webdriver.firefox.options import Options as fx_ops
# from selenium.webdriver.common.keys import Keys
# from html.parser import HTMLParser
from xml.etree.ElementTree import parse
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen as uReq


CHROME_OPTIONS = '--user-agent="Mozilla/5.0 (Windows Phone 10.0; Android 4.2.1; Microsoft; Lumia 640 XL LTE) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Mobile Safari/537.36 Edge/12.10166"'

TASK_ID = 1

EXAMPLE_STAGE_NAMES = {
    1: "webscraper: targetfile",
    2: "webscraper: byxpath",
    3: "webscraper: bylogin",
    4: "webscraper: screenxy"
}



# EXAMPLE-1:
# obtain time-series q/discharge data from a website by constructing download file links
# website link: http://meteo.provincia.bz.it/download-dati.asp#accept-cookies

def webscraper_targetfile(download_dir, q_station, from_year=None, to_year=None):
    # get web driver
    chrome_driver = os.path.join(download_dir, 'chromedriver.exe')
    # define chrome option
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument(CHROME_OPTIONS)
    # create a new chrome session
    # driver = webdriver.Chrome(chrome_driver)
    driver = webdriver.Chrome(executable_path=chrome_driver, chrome_options=chrome_options)
    driver.maximize_window()

    # download data
    for cnt, x in enumerate(q_station):
        print(cnt, x)
        for y in range(from_year, to_year+1):
            print ("\t", y)
            # create download url
            sub_link_1 = 'http://daten.buergernetz.bz.it/services/meteo/v1/timeseries?station_code='
            sub_link_2 = '&output_format=CSV&sensor_code=Q&date_from='
            sub_link_3 = str(y)+'0101'+'0000'+'&date_to='+str(y)+'1231'+'2350'
            url = sub_link_1+str(x)+sub_link_2+sub_link_3
            # download csv file
            driver.get(url)
            time.sleep(5)
            # rename the file in download folder
            flist = os.listdir(download_dir)
            for f in flist:
                if f == 'timeseries':
                    base = os.path.splitext(f)[0]
                    os.rename(
                        os.path.join(download_dir, f),
                        os.path.join(download_dir, base+"_"+str(x)+"_"+str(y)+'.csv')
                    )
    return


# EXAMPLE-2
# obtain time-series q/discharge data from a website using 'find_element_by_xpath' method
# website link: "https://www.arpa.veneto.it/bollettini/storico/Mappa_2019_PORT.htm?t=RG"

def webscraper_byxpath(download_dir, years=None):
    def get_flow_data(outfolder, sn, yr):
        out_path = os.path.join(outfolder, "StationID_"+str(sn)+"_Year_"+str(yr)+".csv")
        w = open(out_path, 'w')
        #w.writelines("Giorno,GEN,FEB,MAR,APR,MAG,GIU,LUG,AGO,SET,OTT,NOV,DIC\n")
        for i in range(1, 33):
            x_path = '//*[@id="meteostorico"]/table[2]/tbody/tr['+str(i)+']'
            element = driver.find_element_by_xpath(x_path).text
            element = element.replace('   ', ' -999 ').split()
            #print (element)
            w.writelines(",".join(str(k) for k in element)+"\n")
        w.close()
        return

    # open xml page where station ids are available
    xmlpage = urllib.request.urlopen("https://www.arpa.veneto.it/bollettini/storico/IDX_2019_PORT.xml")
    xmldoc = parse(xmlpage)
    # get station id for flow/discharge data downloading
    # https://www.foxinfotech.in/2019/04/python-how-to-read-xml-from-url.html
    stations = list()
    for item in xmldoc.iterfind('STAZIONE'):
        title = item.findtext('IDSTAZ')
        stations.append(title)
        
    # get web driver
    # https://www.techbeamers.com/selenium-webdriver-python-tutorial/
    chrome_driver = os.path.join(download_dir, 'chromedriver.exe')
    # define chrome option
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument(CHROME_OPTIONS)
    # create a new chrome session
    # driver = webdriver.Chrome(chrome_driver)
    driver = webdriver.Chrome(executable_path=chrome_driver, chrome_options=chrome_options)
    driver.maximize_window()

    # find and download data
    for s in stations: 
        print (s)
        sid = None
        if len(s) == 1:
            sid = '000'+s
        elif len(s) == 2:
            sid = '00'+s
        elif len(s) == 3:
            sid = '0'+s
        else:
            sid = s
        # loop over years
        for y in years:
            print("\t", y)
            main_url = "https://www.arpa.veneto.it/bollettini/storico/"+str(y)+"/"+str(sid)+"_"+str(y)+"_PORT.htm"
            driver.get(main_url)
            time.sleep(5)
            try:
                get_flow_data(download_dir, s, y)
            except:
                print("\t No Data : ", y)
            time.sleep(1)
            # Get back to google home page and then open next new page
            driver.get("https://www.google.com/")
            time.sleep(2)
    return
    

# EXAMPLE-3
# obtain global dem data from a website by login the page
# login page: https://urs.earthdata.nasa.gov

def webscraper_bylogin(download_dir, data_urls, user_name=None, user_password=None):
    # get web driver
    chrome_driver = os.path.join(download_dir, 'chromedriver.exe')
    # define chrome option
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument(CHROME_OPTIONS)
    # create a new chrome session
    # driver = webdriver.Chrome(chrome_driver)
    driver = webdriver.Chrome(executable_path=chrome_driver, chrome_options=chrome_options)
    driver.implicitly_wait(30)
    driver.maximize_window()
    driver.get('https://urs.earthdata.nasa.gov')
    driver.find_element_by_name("username").send_keys(user_name)
    driver.find_element_by_name("password").send_keys(user_password)
    driver.find_element_by_xpath('//*[@id="login"]/p[5]/input').click()

    # open another new tab
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    # loop over urls for downloading data
    for r in data_urls:
        time.sleep(5)
        r = re.sub('\n', '', r)
        init = r.split("//")[0].split(":")[0]+'s:'+r.split("//")[0].split(":")[1]
        main_url = init+"//"+r.split("//")[1]+"/"+r.split("//")[2]
        driver.get(main_url)
        time.sleep(30)
        driver.close()
        time.sleep(5)
        driver.quit()
    return


# EXAMPLE-4
# obtain html data by given key-value information

def webscraper_product_reviews1(url: object, n: object) -> object:
    # amazon
    all_reviews = list()
    HEADERS = {
        'user-agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5)'
            'AppleWebKit/537.36 (KHTML, like Gecko)'
            'Chrome/79.0.3945.88 Safari/537.36'
        )
    }
    pgnum = url.index("&pageNumber=")+len("&pageNumber=")
    for p in range(n):
        print("Page "+str(p+1)+"/"+str(n))
        each_url = url[0:pgnum]+str(p+1)
        page = requests.get(each_url, headers=HEADERS)
        page_text = page.text
        soup = bs(page_text)  # html5lib/html.parser is a html parser
        each_pg_revs = soup.find_all('span', {'data-hook': 'review-body'})  #https://medium.com/analytics-vidhya/web-scraping-amazon-reviews-a36bdb38b257
        # loop over each-page to collect reviews
        for each_rev in each_pg_revs:
            all_reviews.append(each_rev.text)
    return all_reviews


def webscraper_product_reviews2(search_string=None):
    # flipkart
    flipkart_url = "https://www.flipkart.com/search?q=" + search_string
    uClient = uReq(flipkart_url)
    flipkartPage = uClient.read()
    uClient.close()
    flipkart_html = bs(flipkartPage, "html.parser")
    bigboxes = flipkart_html.findAll("div", {"class": "_1AtVbE col-12-12"})
    del bigboxes[0:3]
    box = bigboxes[0]
    productLink = "https://www.flipkart.com" + box.div.div.div.a['href']
    prodRes = requests.get(productLink)
    prodRes.encoding='utf-8'
    prod_html = bs(prodRes.text, "html.parser")
    print(prod_html)
    commentboxes = prod_html.find_all('div', {'class': "_16PBlm"})
    filename = search_string + ".csv"
    fw = open(filename, "w")
    headers = "Product, Customer Name, Rating, Heading, Comment \n"
    fw.write(headers)
    reviews = []
    for commentbox in commentboxes:
        try:
            name = commentbox.div.div.find_all('p', {'class': '_2sc7ZR _2V5EHH'})[0].text
        except:
            name = 'No Name'
        try:
            rating = commentbox.div.div.div.div.text
        except:
            rating = 'No Rating'
        try:
            commentHead = commentbox.div.div.div.p.text
        except:
            commentHead = 'No Comment Heading'
        try:
            comtag = commentbox.div.div.find_all('div', {'class': ''})
            custComment = comtag[0].div.text
        except Exception as e:
            print("Exception while creating dictionary: ",e)
        mydict = {
            "Product": search_string,
            "Name": name,
            "Rating": rating,
            "CommentHead": commentHead,
            "Comment": custComment
        }
        reviews.append(mydict)
    return reviews



# EXAMPLE-5
# obtain data by filling web form using screen position
# website link: http://www.arpa.piemonte.it/rischinaturali/accesso-ai-dati/annali_meteoidrologici/annali-meteo-idro/banca-dati-idrologica.html

def webscraper_screenxy():
    def Click_Grid(x, y):
        ctypes.windll.user32.SetCursorPos(x, y)
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)
        
    def get_station_data():
        # Line 1 Position
        Click_Grid(65, 47)  # x, y position of first station in the list within the window
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)  # x, y position for save all years data
        time.sleep(10)
        Click_Grid(1125, 685)  # x, y position for pop-up save
        time.sleep(3)
        # Line 2 Position
        Click_Grid(65, 61)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 3 Position
        Click_Grid(65, 76)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 4 Position
        Click_Grid(65, 92)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 5 Position
        Click_Grid(65, 106)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 6 Position
        Click_Grid(65, 121)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 7 Position
        Click_Grid(65, 137)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 8 Position
        Click_Grid(65, 151)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 9 Position
        Click_Grid(65, 166)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 10 Position
        Click_Grid(65, 182)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        # Line 11 Position
        Click_Grid(65, 196)
        time.sleep(3)
        Click_Grid(170, 255)  # x, y position for Durata table
        time.sleep(3)
        Click_Grid(990, 1020)
        time.sleep(10)
        Click_Grid(1125, 685)
        time.sleep(3)
        return

    # Download data
    while 1:
        get_station_data()
        a = input("Please type 1 to reset the station window: ")
        # scroll the window (station)
        if int(a) == 1:
            # this window has total 11 rows/lines/stations
            for i in range(11):
                Click_Grid(240, 200)
                time.sleep(2)
        else:
            break
    return



# manage download examples
def switch_case(stage_id, stage_name):
    match stage_id:
        case 1:
            print(stage_name)
            # ex1
            download_dir = '/temp'
            q_station = ['08155PG','19850PG','29850PG']
            webscraper_targetfile(
                download_dir,
                q_station,
                from_year=2010,
                to_year=2020
            )
        case 2:
            print(stage_name)
            # ex2
            years = [2010, 2011, 2012]
            webscraper_byxpath(
                download_dir,
                years=years
            )
        case 3:
            print(stage_name)
            # ex3
            data_urls = [
                'http://e4ftl01.cr.usgs.gov//MODV6_Dal_D/SRTM/SRTMGL3.003/2000.02.11/N00E109.SRTMGL3.hgt.zip'
                'http://e4ftl01.cr.usgs.gov//MODV6_Dal_D/SRTM/SRTMGL3.003/2000.02.11/N00E022.SRTMGL3.hgt.zip'
                'http://e4ftl01.cr.usgs.gov//MODV6_Dal_D/SRTM/SRTMGL3.003/2000.02.11/N00E027.SRTMGL3.hgt.zip'
                ]
            webscraper_bylogin(
                download_dir,
                data_urls,
                user_name="<type-user-name>",
                user_password="<type-password>"
            )
        case 4:
            print(stage_name)
            # ex4
            amazon_url = 'https://www.amazon.com/KidKraft-Kids-Study-Desk-Chair-White/product-reviews/B00K3EY9G4/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=1'
            # get reviews list
            reviews_list: object = webscraper_product_reviews1(amazon_url, 35) # total review is 347 and each page review is 10
            print(len(reviews_list))
            raw_reviews = pd.DataFrame({'reviews': reviews_list})
            raw_reviews.shape  # examine dimensions/shape of dataframe.
            raw_reviews.head(10)  # examine first n (i.e 10 in this case) rows of dataframe
            ###
            flipkart_prod_reviews = webscraper_product_reviews2(search_string='iphone 10')
            # ex5
            # open the webpage and then run the below application
            webscraper_screenxy()


if __name__ == '__main__':
    task_ids = list(EXAMPLE_STAGE_NAMES.keys())

    if TASK_ID in task_ids:
        # get task_name
        task_name = EXAMPLE_STAGE_NAMES[TASK_ID]

        # check task_name is string or dictionary
        if isinstance(task_name, str):
            switch_case(TASK_ID, task_name)
        else:
            for task_no in task_name.keys():
                stage_name = task_name[task_no]
                switch_case(task_no, stage_name)

# -*- coding: utf-8 -*---------------------------------
"""
Created on Thu Feb 11 17:36:00 2021

@author: Zeeshan Haleem
"""

#------- Web Scrapping of data from TimesJobs website 1st method--------------

import requests
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq

my_url='https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=python&txtLocation='
uClient= uReq(my_url)
page_html= uClient.read()

page_soup = soup(page_html, 'html.parser')
page_soup

#------------------------------------------------------------------------------------------
import os
# os.getcwd()
# os.chdir('E:\ZePycharm\zee_pycharm_Analysis_codes')
import requests
from bs4 import BeautifulSoup

my_url='https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=python&txtLocation='
html_text= requests.get(my_url).text
soup= BeautifulSoup(html_text, 'html.parser')
# print(html_text)
# print(soup.prettify)



f=open('timesJObData.csv', 'w')
headers= "job_title, company_name, skills, job_desc "
f.write(headers)
    
jobs = soup.find_all('li',class_= 'clearfix job-bx wht-shd-bx')

for job in jobs:
    
        published_date =job.find('span', class_='sim-posted').text.strip()
        if 'few' in published_date:
            job_title= job.find('strong', class_='blkclor').text
            company_name= job.find('h3', class_='joblist-comp-name').text.strip()
            skills= job.find('span', class_='srp-skills').text.strip()
            job_desc= job.find('ul', class_='list-job-dtl clearfix').text.strip()
            
            f.write(job_title +"," + company_name+ ","+ skills+ "," +job_desc+ "\n")
            

            # print(f'''
            #       pub_date        : {published_date}
            #       Job_title       : {job_title}
            #       Company Name    : {company_name}
            #       Required Skills : {skills}
            #       Job_description : {job_desc}
                 
            # ''')
f.close()

#-----------------timesJob data scrappin by using 2nd method------------------


from bs4 import  BeautifulSoup as soup
import requests

my_url= "https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=python&txtLocation="

html_text= requests.get(my_url).content
text_soup= soup( html_text, 'html.parser')

containers= text_soup.findAll('li', {"class":"clearfix job-bx wht-shd-bx"})

for container in containers:
    job_title= container.findAll('strong', {"class":"blkclor"})[0].text
    company_name= container.findAll('h3',{"class":"joblist-comp-name"})[0].text
    
    print(f'''Job_title :{job_title}
          company       : {company_name}
          ''')
    
    


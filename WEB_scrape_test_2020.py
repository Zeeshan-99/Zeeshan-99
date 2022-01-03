




# ---------7th-March-2020 (html table scrapping)  100% working-----------------------

from bs4 import BeautifulSoup
import requests
import pandas as pd

url='https://en.wikipedia.org/wiki/List_of_largest_manufacturing_companies_by_revenue'

response= requests.get(url)
soup=BeautifulSoup(response.text,'html.parser')

table= soup.find('table',class_='wikitable').tbody


#-------------------------------------------------
rows= table.find_all('tr')

#---------column: list comprehension method --------------------------

columns=[v.text.replace('\n','') for v in rows[0].find_all('th')]
#---------------column: for loop method -----------------

col_data=[]
for w in rows[0].find_all('th'):
    col_data.append(w.text.strip())

print(col_data)

#------------------Now data storing into data frame-----------

data=[]

rows = table.find_all( 'tr' )
for row in rows:
    cols = row.find_all("td")
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])        # Get rid of empty values



#------------------------------------------------------------------

df=pd.DataFrame(data=data)

df.to_csv('haider.csv',index=False)

#---------------------------------------------------------------------

#---------100% working- Link: https://pythonprogramminglanguage.com/web-scraping-with-pandas-and-beautifulsoup/

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate



res = requests.get("http://www.nationmaster.com/country-info/stats/Media/Internet-users")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0]
df = pd.read_html(str(table))
print( tabulate(df[0], headers='keys', tablefmt='psql') )


#--------- storing as json formate ---------------
res = requests.get("http://www.nationmaster.com/country-info/stats/Media/Internet-users")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0]
df = pd.read_html(str(table))
print(df[0].to_json(orient='records'))
#-------------------------------------------------

#-------- 2nd table extracting------its 100% working -------------------
res = requests.get("https://pythonprogramming.net/parsememcparseface/")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0]
df = pd.read_html(str(table))
print( tabulate(df[0], headers='keys', tablefmt='psql') )



#--------------------------------100% working---2020-Code-----------------------------------------------------------
#----------------------------link: https://www.pluralsight.com/guides/extracting-data-html-beautifulsoup-----------------

# importing the libraries
from bs4 import BeautifulSoup
import requests
import csv

# Step 1: Sending a HTTP request to a URL
url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text

# Step 2: Parse the html content
soup = BeautifulSoup(html_content, "lxml")
# print(soup.prettify()) # print the parsed data of html


# Step 3: Analyze the HTML tag, where your content lives
# Create a data dictionary to store the data.
data = {}
# Get the table having the class wikitable
gdp_table = soup.find("table", attrs={"class": "wikitable"})
gdp_table_data = gdp_table.tbody.find_all("tr")  # contains 2 rows

# Get all the headings of Lists
headings = []
for td in gdp_table_data[0].find_all("td"):
    # remove any newlines and extra spaces from left and right
    headings.append(td.b.text.replace('\n', ' ').strip())

# Get all the 3 tables contained in "gdp_table"
for table, heading in zip(gdp_table_data[1].find_all("table"), headings):
    # Get headers of table i.e., Rank, Country, GDP.
    t_headers = []
    for th in table.find_all("th"):
        # remove any newlines and extra spaces from left and right
        t_headers.append(th.text.replace('\n', ' ').strip())

    # Get all the rows of table
    table_data = []
    for tr in table.tbody.find_all("tr"):  # find all tr's from table's tbody
        t_row = {}
        # Each table row is stored in the form of
        # t_row = {'Rank': '', 'Country/Territory': '', 'GDP(US$million)': ''}

        # find all td's(3) in tr and zip it with t_header
        for td, th in zip(tr.find_all("td"), t_headers):
            t_row[th] = td.text.replace('\n', '').strip()
        table_data.append(t_row)

    # Put the data for the table with his heading.
    data[heading] = table_data

# Step 4: Export the data to csv
"""
For this example let's create 3 seperate csv for
3 tables respectively
"""
for topic, table in data.items():
    # Create csv file for each table
    with open(f"{topic}.csv", 'w') as out_file:
        # Each 3 table has headers as following
        headers = [
            "Country/Territory",
            "GDP(US$million)",
            "Rank"
        ]  # == t_headers
        writer = csv.DictWriter(out_file, headers)
        # write the header
        writer.writeheader()
        for row in table:
            if row:
                writer.writerow(row)




#------------------------------Link :https://medium.com/analytics-vidhya/web-scraping-wiki-tables-using-beautifulsoup-and-python-6b9ea26d8722---

import requests

url="https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area"

website_url=requests.get(url).text

from bs4 import BeautifulSoup
soup = BeautifulSoup(website_url,"lxml")
print(soup.prettify())


My_table = soup.find("table",{"class":"wikitable sortable"})

links= My_table.findAll("a")


Countries=[]

for link in links:
     Countries.append(link.get('title'))

import  pandas as pd

df=pd.DataFrame()
df['Countries']=Countries

#------------------------------------------------------------------------------------------------------------

#-------100% working-- link:https://pythonprogramming.net/tables-xml-scraping-parsing-beautiful-soup-tutorial/-----------------

import pandas as pd
from bs4 import BeautifulSoup
import requests
import csv


source = requests.get('https://pythonprogramming.net/parsememcparseface/').text
soup = BeautifulSoup(source,'lxml')


data = []
table = soup.find("table")
# table_body= table.find("tbody")

rows = table.findAll('tr')
for row in rows:
    cols = row.findAll("td")
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele]) # Get rid of empty values

df=pd.DataFrame(data)

#-------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------100% working--link: https://news.codecademy.com/web-scraping-python-beautiful-soup-mlb-stats/


import  pandas as pd
import  re              # regular expression
import requests
from bs4 import BeautifulSoup


page= requests.get('http://www.espn.com/mlb/history/leaders/_/breakdown/season/year/2018/start/1')
soup= BeautifulSoup(page.text,'lxml')
print(soup.prettify())

#-----------Column----------------
header=soup.find('tr',class_="colhead")
columns=[v.text for v in header.find_all('td')]

#------------Now creating an empty dataframe from header ----------------------------------------------------------------------------
final_df=pd.DataFrame(columns=columns)
#----------------------------------------------------
#------------Now finding all values from table except header----------------------------------------------------------------------
# players=soup.find_all('tr',class_=re.compile('row player-10-'))

players=soup.find_all('tr',attrs={'class':re.compile('player-10')})


stats=[]
for p in players:
    cols=p.find_all('td')
    cols=[v.text.strip() for v in cols]
    stats.append([q for q in cols if q])



temp_df = pd.DataFrame(stats)
print(temp_df)


# create a dataframe for the single player's stats

temp_df.columns=columns

#------------------------Joining -----------------------------------------------------------------------------------------------------
final_df=pd.concat([final_df,temp_df],ignore_index=True)

final_df.to_csv('player.csv',index=False)




#------------------------------------------------------------------------------------------------------------------------------------
###----100% working -Flipkart data scrapping- https://github.com/Bijay555/flipkart-Web-Scraping-using-python  --------------------------


from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq

my_url="https://www.flipkart.com/search?q=samsung+mobiles&sid=tyy%2C4io&as=on&as-show=on&otracker=AS_QueryStore_HistoryAutoSuggest_0_2&otracker1=AS_QueryStore_HistoryAutoSuggest_0_2&as-pos=0&as-type=HISTORY&as-searchtext=sa"

uClient = uReq(my_url)
page_html = uClient.read()
uClient.close()
page_soup = soup(page_html, "html.parser")

containers = page_soup.findAll("div", { "class": "_3O0U0u"})
#print(len(containers))

#print(soup.prettify(containers[0]))

container = containers[0]
#print(container.div.img["alt"])

price = container.findAll("div", {"class": "col col-5-12 _2o7WAb"})
#print(price[0].text)


ratings = container.findAll("div", {"class": "niH0FQ"})
#rint(ratings[0].text)

filename = "products.csv"
f = open(filename, "w")

headers = "Product_Name, Pricing, Ratings \n"
f.write(headers)

for container in containers:
    product_name = container.div.img["alt"]
    price_container = container.findAll("div", {"class": "col col-5-12 _2o7WAb"})
    price = price_container[0].text.strip()

    rating_container = container.findAll("div", {"class": "niH0FQ"})
    rating = rating_container[0].text

   #rint("Product_Name:"+ product_name)
   #print("Price: " + price)
   #print("Ratings:" + rating)

    #String parsing
    trim_price=''.join(price.split(','))
    rm_rupee = trim_price.split('₹')
    add_rs_price = "Rs."+rm_rupee[1]
    split_price = add_rs_price.split('E')
    final_price = split_price[0]

    split_rating = rating.split(" ")
    final_rating = split_rating[0]

    print(product_name.replace("," ,"|") +"," + final_price +"," + final_rating + "\n")
    f.write(product_name.replace("," ,"|") +"," + final_price +"," + final_rating + "\n")
f.close()
#----------------------------------------------------------------------------------------------------

import requests
from bs4 import BeautifulSoup

scrape_flipkart('http://www.flipkart.com/watches/pr?p%5B%5D=facets.ideal_for%255B%255D%3DMen&p%5B%5D=sort%3Dpopularity&sid=r18&facetOrder%5B%5D=ideal_for&otracker=ch_vn_watches_men_nav_catergorylinks_0_AllBrands', 5)


def scrape_flipkart(url, no_products):
    r = requests.get(url)
	soup = BeautifulSoup(r.content)
	data = soup.find_all("div", {"class":"product-unit unit-3 browse-product quickview-required"})
    product_name = []
    image_url = []
    price = []
    link = []
    for item in data:
        name = item.find_all("a",{"class":"fk-display-block"})[0]
        product_name.append(name.get("title"))
        image = item.find_all("img")[0]
if image.get("data-src"):
            img_url = image.get("data-src")
        else:
            img_url = image.get("src")
        image_url.append(img_url)
        price1 = item.find_all("div",{"class":"pu-final font-dark-color fk-inline-block"})[0]
        price.append(price1.text.strip())
        link.append(name.get("href"))
product_name_final = product_name[:no_products]
    image_url_final = image_url[:no_products]
    price_final = price[:no_products]
    link_final = link[:no_products]


#------------------------------------------------------------------------------
#-------------100% working ---flipkart search --and scraping all data-----

#--------------link: https://github.com/Rakshith-V/Web-scraping-flipkart-using-beautiful-soup/blob/master/Mobile%20data%20scraping%20scraping%20with%20beautiful%20soup/old/scrap.py
import requests
from bs4 import BeautifulSoup
url = "https://www.flipkart.com/search?q="

query = input("Enter the product you want to search for: ")
#replacing all spaces with %20 sign.
query = query.replace(" ","%20")
url = url+query

source = requests.get(url).content
soup = BeautifulSoup(source,'html.parser')

#extracting via tags
names = soup.findAll('div',{'class':'_3wU53n'})
rating = soup.findAll("div",{"class":"niH0FQ"})
price = soup.findAll("div",{"class":"_1vC4OE _2rQ-NK"})
for i,j,k in zip(names,rating,price):
	print(i.text.strip(),'\n',j.text,'\n',k.text.strip(),'\n')
	print('\n')

#---------------------------2nd method-------------------------------------


import pandas as pd
import requests
from bs4 import BeautifulSoup
url = "https://www.flipkart.com/search?q="

query = input("Enter the product you want to search for: ")
#replacing all spaces with %20 sign.
query = query.replace(" ","%20")
url = url+query

source = requests.get(url).content
soup = BeautifulSoup(source,'html.parser')


Name=[]
Rating=[]
Price=[]
#extracting via tags
names = soup.findAll('div',{'class':'_3wU53n'})
rating = soup.findAll("div",{"class":"niH0FQ"})
price = soup.findAll("div",{"class":"_1vC4OE _2rQ-NK"})

for i,j,k in zip(names,rating,price):

    name_l=[p for p in i]
    Name.append(name_l)

    rating_l=[q.text for q in j]
    Rating.append(rating_l)

    price_l=[r.strip() for r in k]
    Price.append(price_l)


df=pd.DataFrame({'Name':Name,'Rating':Rating,'Price':Price})

df.to_csv('farhan.csv')


#--------------------------------------------------------------------------------------------------------------------
#---------------100% (working)- nCovid19- html table scrapping and cleaning with for loop -----(22-Apr-2020)-------------

#------------ data link: https://www.worldometers.info/coronavirus/
#------------ code link:https://www.kaggle.com/ukveteran/web-scraping-live-covid-19-data-jma/notebook

import pandas as pd
import numpy as np
import json, requests
from bs4 import BeautifulSoup

#------------------Html Table scrapping using BeautifulSoup ------------------------------------------------------------------------

URL = 'https://www.worldometers.info/coronavirus/' #the website the data is extracted

page= requests.get(URL)
soup=BeautifulSoup(page.text,'lxml')

table = soup.find('table',{"class": 'table'})

rows = table.find_all('tr')

#-------------fist we scrapping header column from table and there are two method-----------------

#--------1st method (list comprehensive)- only header scrapping from html table-------------------------

head=[v.text.replace('\n','') for v in rows[0].find_all('th')[:12]]


#-------- 2nd method (looping method) ---- Only header scrapping from html table-------------------------
head=[]
for w in rows[0].find_all('th')[:12]:
    head.append(w.text.strip())

print(head)

#------------------------- Now data except header column scrapping from html table -------------------------

data=[]
rows = table.find_all('tr')                  # we see all table row

for row in rows[8:221]:                      # we start scrapping data from 8th_row to 221th_row
    cols= row.find_all('td')[:12]              # here we scrapping the data only upto 3 column for each row
    cols=[k.text.strip() for k in cols ]
    data.append(cols)                              # to append data


#---------Making dataframe from the scrapped data -------------------------------------------------------------------------------------------

df=pd.DataFrame(data, columns=head)
print(df)
#------------Now Data Cleanning using for loop: removing +/- sing from table -----------------------------------------------------------------------------------------------------------------------------------------

def dataframeCleaner(df):
    for columnname in df:  # looping through titles of the table
        temp = []
        for column in df[columnname]:  # geting column elements for the each title
            column = str(column)
            column = column.replace(',', '')  # Removing unwanted data clutter
            column = column.replace('+', '')  # Removing unwanted '+'sign
            try:  # using try except block to convert datatype string to integer while avoiding error
                column = int(column)
            except:
                pass

            temp.append(column)
        df[columnname] = temp

    df = df.drop(df.tail(1).index)  # Deleting the last row
    df = df.replace(r'^\s*$', 0, regex=True)  # converting empty string to 0
    return df

df1=dataframeCleaner(df)
print(df1)

#----------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
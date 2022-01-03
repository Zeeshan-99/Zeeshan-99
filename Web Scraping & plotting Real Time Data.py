#------------------ 1st we go for real live data --100% working ------------------------------------------------------------

# importing libraries
import time
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime



def real_time_price(stock_code):
    url=("https://in.finance.yahoo.com/quote/")+stock_code+('.HK?p=')+stock_code+('.HK&.tsrc=fin-srch')
    r= requests.get(url)
    web_content = BeautifulSoup(r.text,'lxml')
    web_content = web_content.find('div', {"class":'My(6px) Pos(r) smartphone_Mt(6px)'})
    web_content = web_content.find('span').text

    if web_content == []:
        web_content = '999999'

    return web_content

# web_content=real_time_price('0001')
# print(web_content)

HSI=['0001','0002','0003','0005']
for step in range(1,201):
    price=[]
    col=[]
    time_stamp=datetime.datetime.now()
    time_stamp=time_stamp.strftime("%Y-%m-%d %H:%M:%S")

    for stock_code in HSI:
        price.append(real_time_price(stock_code))

    while True:
        col = [time_stamp]
        col.extend(price)
        with open('zeeStock_yahoo.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(col)
            print(col)

        time.sleep(75)


    # df=pd.DataFrame(col)
    # df=df.T
    # df.to_csv('zeeStock_yahoo.csv', mode='a',header=False)
    # print(col)

#-------------Now live data visualization with the real data ------------------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style

style.use('fivethirtyeight')

fig= plt.figure()
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)



def animate(i):
    df=pd.read_csv('zeeStock_yahoo.csv')
    ys=df.iloc[1:,2].values
    xs=list(range(1,len(ys)+1))
    ax1.clear()
    ax1.plot(xs,ys)
    ax1.set_title('CKH Holdings',fontsize=12)

    ys = df.iloc[1:,3].values
    ax2.clear()
    ax2.plot(xs, ys)
    ax2.set_title('CKH Holdings', fontsize=12)

    ys = df.iloc[1:,4].values
    ax3.clear()
    ax3.plot(xs, ys)
    ax3.set_title('CKH Holdings', fontsize=12)

    ys = df.iloc[1:,5].values
    ax4.clear()
    ax4.plot(xs, ys)
    ax4.set_title('HSBC Holdings', fontsize=12)

ani= FuncAnimation(fig,animate,interval=10000)
plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------------------------------------------

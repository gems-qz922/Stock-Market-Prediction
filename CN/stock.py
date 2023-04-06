import requests 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sqlite3
from datetime import datetime

class Stock:
    def __init__(self, symbol, lmt=5000):
        '''symbol: string of the 6 digit number of a stock
        lmt: limit of length of data, default 5000'''
        self.symbol = symbol
        self.lmt = lmt
        self.df = None
        self.db_setup()


    def db_setup(self):
        self.connection = sqlite3.connect("test.sqlite3") # set up db
        self.cursor = self.connection.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS CN_stock_prices (
                        symbol TEXT,
                        date DATETIME,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        PRIMARY KEY (symbol, date)
                        );""")
        self.connection.commit()


    def store_data(self):
        parsed_data = [(self.symbol,) + tuple(row.split(',')[0:6]) for row in self.info]
        # Insert the data into the database
        self.cursor.executemany(
            """INSERT OR IGNORE INTO CN_stock_prices (symbol, date, open, close, high, low, volume) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            parsed_data
        )
        self.connection.commit()


    def get_data(self):
        '''The function to get historic price information of a stock.
        symbol: string of the 6 digit number of a stock'''
        query = f"""SELECT * FROM CN_stock_prices WHERE symbol = '{self.symbol}' ORDER BY date ASC"""
        self.df = pd.read_sql_query(query, con = self.connection)
        last_date = datetime.strptime(self.df['date'].iloc[-1], '%Y-%m-%d')
        if (datetime.now() - last_date).days >= 1: # if not up to date, re-scrapy data
            code = '0.' + self.symbol if self.symbol[0] == '0' else '1.' + self.symbol    
            data = requests.get(f'https://61.push2his.eastmoney.com/api/qt/stock/kline/get?secid={code}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101&smplmt={self.lmt}&lmt=1000000&_=1673210300985')
            # print(info.json()['data'].keys()) --> dict_keys(['code', 'market', 'name', 'decimal', 'dktotal', 'preKPrice', 'klines'])
            self.info = data.json()['data']['klines'] # change str to json, and choose the inner dictionary
            self.store_data()
            query = f"""SELECT * FROM CN_stock_prices WHERE symbol = '{self.symbol}' ORDER BY date ASC"""
            self.df = pd.read_sql_query(query, con = self.connection)
        return self.df
        

    def to_csv(self,**args): # **args eg. header=False
        if not self.df:
            self.get_data()
        self.df.to_csv(f'{self.symbol}.csv',index=False,**args)


    def plot_klines(self):
        # check if self.df exists
        if not self.df:
            self.df = self.get_data()
        plt.plot(self.df['date'],self.df['close'])
        plt.xticks(np.arange(0, len(self.df['date']) + 1, 300))
        plt.show()

# test if it is working 
if __name__ == '__main__':
    matplotlib.use('TkAgg')
    s = stock('002230')
    s.plot_klines()

import yfinance as yf
import pandas as pd
import numpy as np
import os

def get_yfinance_data(start_year, tickers):
    
    prices = yf.download(tickers, start=f"{start_year}-01-01", end='2023-12-31')['Adj Close'].dropna(axis=1)
    prices.index = pd.to_datetime(prices.index)

    stocks = np.log(prices/prices.shift(1)).dropna()
    stocks.to_csv(f'data/stocks-{start_year}.csv')

    prices = yf.download(['^GSPC'], start=f"{start_year}-01-01", end='2023-12-31')['Adj Close']
    prices.index = pd.to_datetime(prices.index)

    stocks = np.log(prices/prices.shift(1)).dropna()
    stocks.to_csv(f'data/mkt-{start_year}.csv')



  

if __name__ == "__main__":
    print(os.listdir(os.getcwd()))
    tickers = pd.read_csv('data/tickers-2005.csv', header=None)
    get_yfinance_data(2005, tickers.iloc[:,0].tolist())
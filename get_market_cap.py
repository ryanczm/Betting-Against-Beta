import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import numpy as np


def process_market_cap(market_cap_str):
    # Remove whitespace and strip the dollar sign
    market_cap_str = market_cap_str.strip().lstrip('$')

    # Get the last character (T, B, or M) to determine the unit
    unit = market_cap_str[-1]

    # Remove the unit character to get the numeric value
    numeric_value = float(market_cap_str[:-1])

    # Convert to millions based on the unit
    if unit == 'T':
        return numeric_value * 1e6
    elif unit == 'B':
        return numeric_value * 1e3
    elif unit == 'M':
        return numeric_value
    else:
        return None  # Handle invalid units


def scrape_market_cap(url):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        if len(tables) > 0:
            target_table = tables[0]  # Assuming the market cap table is the first table
            years = []
            market_caps = []
            for row in target_table.find_all('tr'):
                columns = row.find_all('td')
                if len(columns) == 3:
                    year = columns[0].text.strip()
                    market_cap_str = columns[1].text.strip()
                    market_cap_str = process_market_cap(market_cap_str)
                    market_caps.append(market_cap_str)
                    years.append(year)
            return pd.Series(market_caps, index=years, name='Market Cap')
    return None
    

# Function to fetch market cap data for a list of tickers
def fetch_market_cap_data(tickers_df, tickers_to_name_df):
    # Initialize an empty DataFrame to store the market cap data
    market_cap_df = pd.DataFrame()

    # Iterate through tickers and fetch market cap data
    for ticker in tickers_df['ticker']:
        # Find the corresponding company name
        company_name = tickers_to_name_df.loc[tickers_to_name_df['Symbol'] == ticker, 'Company Name'].values[0]

        # Generate the URL
        company_url = f"https://companiesmarketcap.com/{company_name}/marketcap/"

        # Scrape market cap data
        market_cap_series = scrape_market_cap(company_url)

        if market_cap_series is not None:
            # Add the market cap series to the market_cap_df
            market_cap_df[ticker] = market_cap_series

        # Sleep to avoid overloading the website (adjust the sleep duration as needed)
        time.sleep(random.uniform(1, 3))  # Sleep for 2 seconds between requests

    return market_cap_df

# Read 'tickers-2005.csv' to get a list of tickers
tickers_df = pd.read_csv('data/tickers-2005.csv')

# Read 'tickers-to-name.csv' to get mapping of tickers to company names
tickers_to_name_df = pd.read_csv('data/tickers-to-name.csv')

market_cap_df = fetch_market_cap_data(tickers_df, tickers_to_name_df)


pd.DataFrame(market_cap_df.columns).to_csv('data/ticks.csv')
market_cap_df.to_csv('data/market-cap-df.csv')
market_cap_df.to_pickle('data/market-cap-df.pkl')
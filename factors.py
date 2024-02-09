import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats


def calc_stock_beta(stock_returns, market_returns):
    """ Used to show capital market line too flat, Black 1972"""
    market_returns_with_const = sm.add_constant(market_returns)
    model = sm.OLS(stock_returns, market_returns_with_const).fit()
    beta = model.params[1]
    return beta

def calc_fp_betas(stock_returns, market_returns):
    """apply, column wise"""
    stock_returns_3d = stock_returns.rolling(3).sum().dropna()
    market_returns_3d = market_returns.rolling(3).sum().dropna() 
    
    corr = stock_returns_3d.rolling(252*5).corr(market_returns_3d).dropna()
    stock_vol = stock_returns.rolling(252).std().dropna()
    market_vol = market_returns.rolling(252).std().dropna()
    
    betas = corr.mul(stock_vol,axis=0).div(market_vol,axis=0)
    betas = betas.reindex(stock_returns.index)
    return betas['Adj Close']


def shrink_fp_betas_cross_sectionally(beta):
    """applymap, element wise"""    
    return 0.6 * beta + 0.4 * 1


def process_ranked_beta_row(row):
    # Step 1: Rank the betas
    ranked_betas = row.rank()
    # Step 2: Calculate row-wise median rank
    median_rank = np.median(ranked_betas)
    # Step 3: Subtract each rank by the median rank
    rank_minus_median = ranked_betas - median_rank
    # Step 4: Calculate row-wise sum of rank minus median rank for stocks above median rank. +ve. low beta
    sum_rank_minus_median_above = rank_minus_median[rank_minus_median > 0].sum()
    # Step 5: Calculate row-wise sum of rank minus median rank for stocks below median rank. -ve. high beta
    sum_rank_minus_median_below = rank_minus_median[rank_minus_median < 0].sum()
    # Step 6: Divide each stock above median rank by the calculated sum from Step 4
    for i in range(len(rank_minus_median)):
        # long low beta
        if rank_minus_median[i] >= 0:
            rank_minus_median[i] /= sum_rank_minus_median_above
        # Short high beta
        if rank_minus_median[i] < 0:
            rank_minus_median[i] /= -1*sum_rank_minus_median_below
    return rank_minus_median
        
def process_ew_beta_row(row):
    # Step 1: Rank the betas
    ranked_betas = row.rank()

    # Step 2: Calculate row-wise median rank
    median_rank = np.median(ranked_betas)

    # Step 3: Subtract each rank by the median rank
    rank_minus_median = ranked_betas - median_rank

    # Step 4: Calculate size
    size = (len(row/2))

    # Step 5: Assign weights
    for i in range(len(rank_minus_median)):
        # low beta, long 
        if rank_minus_median[i] >= 0:
            rank_minus_median[i] = 1/size
        # high beta, short
        if rank_minus_median[i] < 0:
            rank_minus_median[i] = -1/size
    return rank_minus_median

def process_value_beta_row(row, market_cap_df):
    
    market_cap = market_cap_df.loc[pd.to_datetime(f"{row.name.year}-01-01")]
    
    # Step 1: Rank the betas
    ranked_betas = row.rank()

    # Step 2: Calculate row-wise median rank
    median_rank = np.median(ranked_betas)

    # Step 3: Subtract each rank by the median rank
    rank_minus_median = ranked_betas - median_rank

    # Step 4: Calculate row-wise sum of rank minus median rank for stocks above median rank. +ve. low beta
    sum_rank_minus_median_above = rank_minus_median[rank_minus_median > 0].sum()
    market_cap_above = market_cap[row.index[rank_minus_median>0]].sum()
    # print(market_cap_above)
    # Step 5: Calculate row-wise sum of rank minus median rank for stocks below median rank. -ve. high beta
    sum_rank_minus_median_below = rank_minus_median[rank_minus_median < 0].sum()
    market_cap_below = market_cap[row.index[rank_minus_median<0]].sum()
    # print(market_cap_below)

    # Step 6: Divide each stock above median rank by the calculated sum from Step 4
    for i in range(len(rank_minus_median)):
        # long low beta
        if rank_minus_median[i] >= 0:
            rank_minus_median[i] /= (1/market_cap[i]) * sum_rank_minus_median_above * market_cap_above 
        # Short high beta
        if rank_minus_median[i] < 0:
            rank_minus_median[i] /= (1/market_cap[i]) * -1*sum_rank_minus_median_below * market_cap_below

    return rank_minus_median
       

def calc_rank_weights(daily_betas):
    """dataframe of betas of universe"""    
    monthly_betas = daily_betas.resample('M').last()
    return monthly_betas.apply(process_ranked_beta_row, axis=1)
   
def calc_equal_weights(daily_betas):
    """dataframe of betas of universe"""    
    monthly_betas = daily_betas.resample('M').last()
    return monthly_betas.apply(process_ew_beta_row, axis=1)

def calc_value_weights(daily_betas, market_cap_df):
    """dataframe of betas of universe"""    
    monthly_betas = daily_betas.resample('M').last()
    return monthly_betas.apply(process_value_beta_row, market_cap_df=market_cap_df, axis=1)


def calc_ls_returns(lp, sp, rets):
    """combine long and short portfolio weights to get returns"""   
    lrets = lp.mul(rets).dropna().sum(axis=1)
    srets = sp.mul(rets).dropna().sum(axis=1)
    return lrets, srets, lrets.sub(srets)   


def plot_cml(df):
    # Scatterplot with regression line
    df.plot(kind='scatter',x='betas', y='returns', xlim=[0,3], ylim=[-0.5,2.5], marker='x')
    # sns.regplot(x='betas', y='returns', data=df, scatter_kws={'s': 100})

    # Fit a linear regression model
    X = df['betas'].values.reshape(-1, 1)
    y = df['returns'].values
    model = LinearRegression().fit(X, y)
    beta_coef = model.coef_[0]
    beta_std_error = stats.sem(y - model.predict(X))
    t_statistic = beta_coef / beta_std_error
    r_squared = model.score(X, y)

    # Plot the observed CML line
    plt.plot([0, 3], [model.intercept_, model.intercept_+  3*model.coef_[0]], label='Observed CML', color='red')

    # Plot the CAPM CML line (slope 1)
    plt.plot([0, 3], [0.03, 2], label='CAPM CML', color='magenta')

    # Display the regression equation on the plot
    plt.annotate(f'E[Ri]-Rf = {model.intercept_:.2f} + {model.coef_[0]:.2f} * (Rm-Rf)\n t-stat: {t_statistic:.2f}\n R2: {r_squared:.2f}', fontsize=10, xy=(0.02, 0.87), xycoords='axes fraction', bbox=dict(facecolor='lightgrey', edgecolor='grey', boxstyle='round'),
                 color='black')

    # Set labels and legend
    plt.xlabel('Betas')
    plt.ylabel('Mean excess return (%)')
    plt.legend()
    plt.title('Observed CML vs Theoretical CML')

    # Show the plot
    plt.show()

def create_ff_bab_df(bab, ff):
    factor_df = ff.iloc[:,:-1][ff.index.year >= 2010]
    factor_df['BAB'] = bab.values
    # factor_df.iloc[0] = factor_df.iloc[0] + 1
    return factor_df

def create_mkt_bab_df(bab, mkt):
    mkt = mkt[mkt.index.year >= 2010]
    factor_df = pd.DataFrame({"BAB":bab.values.flatten(),"MKT":mkt.values.flatten()}, index=bab.index)
    # factor_df.iloc[0] = factor_df.iloc[0] + 1
    return factor_df
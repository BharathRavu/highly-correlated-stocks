### Highly correlated indian stocks


```python
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import pandas as pd
import os
import numpy as np
import time
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
file_path=os.path.join('data','top_100_bse.txt')
file1 = open(file_path, 'r') 
Lines = file1.readlines() 
#top_100=pd.DataFrame()
file_path2=os.path.join('data','Equity.csv')
bse_cmpns=pd.read_csv(file_path2)
security_id=[]
security_code=[]
market_cap=[]
for line in Lines:
    line=line.replace('\n', "").replace('\t', " ")
    line=line.split()
    security_code.append(line[0])
    indx=bse_cmpns[bse_cmpns['Security Code']== int(line[0])].index.tolist()
#    print(line[0])
#    print(indx)
#    print(bse_cmpns['Security Id'][indx[0]])
    security_id.append(bse_cmpns['Security Id'][indx[0]])
    market_cap.append(float(line[-1]))

arr1=np.array(security_code)
arr2=np.array(security_id)
arr3=np.array(market_cap)
arr = np.vstack((arr1, arr2, arr3))           
top_100 = pd.DataFrame(np.transpose(arr), columns = ['Security Code', 'Security Id', 'Market Cap in Cr.']) 
#top_100['Security Code']=top_100['Security Code'].astype(int)
top_100.to_csv(r'data\top100.csv',index=False)
```

```python
start_date = '2019-10-07'
end_date = '2020-10-07'
```


```python
ts = TimeSeries(key='6LG1CFT7X2M2ZESX', output_format='pandas')
```


```python
def get_daily_stock_prices_alpha_vantage_BSE(ts, token, start_date, end_date, variables):
    """
    ts = TimeSeries(key='6LG1CFT7X2M2ZESX', output_format='pandas) 
    token is the symbol of a company. e.g. 'TCS' or '532540'
    variables are Open, high, low, close, adjusted close, volume, dividend amount, split coefficient
    """
    stock_hist = ts.get_daily_adjusted(symbol='BSE:'+token, outputsize='full')
    stock_hist=stock_hist[0]
    stock_hist.index.names = ['Date']
    stock_hist.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
                               '4. close':'Close', '5. adjusted close': 'Adj Close', 
                               '6. volume': 'Volume' },  inplace=True)
    mask = (stock_hist.index >= start_date) & (stock_hist.index <= end_date)
    return stock_hist.loc[mask][variables]
```


```python
def get_daily_close_prices_top100_companies(ts, top100, start_date, end_date):
    data=pd.DataFrame()
    
    for i in range(top100.shape[0]):
        try:
            S=get_daily_stock_prices_alpha_vantage_BSE(ts, top100['Security Code'][i], start_date, end_date, ['Close'])
            data[top100['Security Id'][i]]=S['Close']
        except ValueError:
            try:
                S=get_daily_stock_prices_alpha_vantage_BSE(ts, top100['Security Id'][i], start_date, end_date, ['Close'])
                data[top100['Security Id'][i]]=S['Close']
            except ValueError:
                print("Could not get data for: "+ top100['Security Id'][i])       
                
        time.sleep(15)

    return data
```


```python
top100_close=get_daily_close_prices_top100_companies(ts, top_100, start_date, end_date)
top100_close.to_csv(r'data\top100_close.csv',index=False)
```

   
    


```python
daily_returns= top100_close.pct_change(1)
daily_returns.dropna(inplace=True)
daily_logreturns=np.log(1+daily_returns)
```


```python
corr_matrix=daily_returns.corr()
```


```python
corr_matrix['TCS']['RELIANCE']
```




    0.5232259150858998




```python
symbols=corr_matrix.columns
len_symbols=len(symbols)
```


```python
# Creating lower triangular matrix indices for pairs
def generate_sorted_corr_all_pairs(daily_returns):
    corr_matrix=daily_returns.corr()
    symbols=corr_matrix.columns
    len_symbols=len(symbols)
    Stocks_row=[]
    Stocks_col=[]
    corr_val=[]
    for i in range(1,len_symbols):
        for j in range(i):
            Stocks_row.append(symbols[i])
            Stocks_col.append(symbols[j])
            corr_val.append(corr_matrix[symbols[i]][symbols[j]])
            
    df=pd.DataFrame()
    arr1=np.array(Stocks_row)
    arr2=np.array(Stocks_col)
    arr3=np.array(corr_val)
    arr3=np.around(arr3, 3)
    arr = np.transpose(np.vstack((arr1, arr2, arr3)))
    df= pd.DataFrame(arr, columns = ['Stocks_X', 'Stocks_Y', 'correlation'])
    df=df.sort_values(by='correlation', ascending=False)
    df=df.reset_index(drop=True)
    return df
```


```python
cor=generate_sorted_corr_all_pairs(daily_returns)
```


```python
cor.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Stocks_X</th>
      <th>Stocks_Y</th>
      <th>correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BAJAJFINSV</td>
      <td>BAJFINANCE</td>
      <td>0.889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AXISBANK</td>
      <td>ICICIBANK</td>
      <td>0.821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACC</td>
      <td>AMBUJACEM</td>
      <td>0.814</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HDFC</td>
      <td>HDFCBANK</td>
      <td>0.798</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TATASTEEL</td>
      <td>JSWSTEEL</td>
      <td>0.797</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HINDALCO</td>
      <td>JSWSTEEL</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BAJAJFINSV</td>
      <td>AXISBANK</td>
      <td>0.772</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ICICIBANK</td>
      <td>HDFC</td>
      <td>0.766</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ICICIBANK</td>
      <td>HDFCBANK</td>
      <td>0.762</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SHREECEM</td>
      <td>ULTRACEMCO</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>10</th>
      <td>KOTAKBANK</td>
      <td>HDFCBANK</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>11</th>
      <td>HINDALCO</td>
      <td>TATASTEEL</td>
      <td>0.749</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BAJFINANCE</td>
      <td>ICICIBANK</td>
      <td>0.741</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NESTLEIND</td>
      <td>HINDUNILVR</td>
      <td>0.735</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SBIN</td>
      <td>ICICIBANK</td>
      <td>0.734</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BAJAJFINSV</td>
      <td>ICICIBANK</td>
      <td>0.733</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PIDILITIND</td>
      <td>ASIANPAINT</td>
      <td>0.731</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GRASIM</td>
      <td>ULTRACEMCO</td>
      <td>0.724</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BERGEPAINT</td>
      <td>ASIANPAINT</td>
      <td>0.722</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ICICIPRULI</td>
      <td>HDFCLIFE</td>
      <td>0.719</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BRITANNIA</td>
      <td>HINDUNILVR</td>
      <td>0.719</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HINDALCO</td>
      <td>ULTRACEMCO</td>
      <td>0.715</td>
    </tr>
    <tr>
      <th>22</th>
      <td>HINDALCO</td>
      <td>VEDL</td>
      <td>0.712</td>
    </tr>
    <tr>
      <th>23</th>
      <td>JSWSTEEL</td>
      <td>ICICIBANK</td>
      <td>0.711</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AXISBANK</td>
      <td>SBIN</td>
      <td>0.711</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

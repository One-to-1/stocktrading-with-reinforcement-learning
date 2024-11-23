# stocktrading-with-reinforcement-learning

Sure, let's go through the `add_indicators` function and explain each of the three technical indicators it adds to the DataFrame.

### Function Definition
```python
def add_indicators(df):
    # 20-day moving average
    df['MA20'] = df['adj_close'].rolling(window=20).mean()
    # RSI
    delta = df['adj_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['adj_close'].ewm(span=12, adjust=False).mean()
    exp2 = df['adj_close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    return df
```

### 1. 20-Day Moving Average (MA20)
```python
df['MA20'] = df['adj_close'].rolling(window=20).mean()
```
- **20-Day Moving Average (MA20)**: This indicator calculates the average of the adjusted closing prices over the past 20 days.
  - `df['adj_close'].rolling(window=20).mean()`: This line creates a rolling window of 20 days over the `adj_close` column and computes the mean for each window.
  - The result is a new column `MA20` in the DataFrame that contains the 20-day moving average.

### 2. Relative Strength Index (RSI)
```python
delta = df['adj_close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
```
- **Relative Strength Index (RSI)**: This momentum oscillator measures the speed and change of price movements, typically used to identify overbought or oversold conditions.
  - `delta = df['adj_close'].diff()`: Calculates the difference between consecutive adjusted closing prices.
  - `gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()`: Isolates positive price changes (gains) and computes their 14-day rolling average.
  - `loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()`: Isolates negative price changes (losses) and computes their 14-day rolling average.
  - `rs = gain / loss`: Calculates the relative strength (RS) as the ratio of average gain to average loss.
  - `df['RSI'] = 100 - (100 / (1 + rs))`: Converts the RS to the RSI using the formula `100 - (100 / (1 + RS))`.

### 3. Moving Average Convergence Divergence (MACD)
```python
exp1 = df['adj_close'].ewm(span=12, adjust=False).mean()
exp2 = df['adj_close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
```
- **Moving Average Convergence Divergence (MACD)**: This trend-following momentum indicator shows the relationship between two moving averages of a security’s price.
  - `exp1 = df['adj_close'].ewm(span=12, adjust=False).mean()`: Calculates the 12-day Exponential Moving Average (EMA) of the adjusted closing prices.
  - `exp2 = df['adj_close'].ewm(span=26, adjust=False).mean()`: Calculates the 26-day EMA of the adjusted closing prices.
  - `df['MACD'] = exp1 - exp2`: Computes the MACD line by subtracting the 26-day EMA from the 12-day EMA.

### Summary
- **MA20**: Smooths out price data to identify the direction of the trend over 20 days.
- **RSI**: Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
- **MACD**: Indicates changes in the strength, direction, momentum, and duration of a trend.

Similar code found with 1 license type
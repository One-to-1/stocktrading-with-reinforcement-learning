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





Sure, let's go through the provided code snippet in detail, explaining each variable and its purpose.

### Code Snippet
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

class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, commission_fee=0.01, slippage_cost=0.1):
        super(StockTradingEnv, self).__init__()
        self.data = add_indicators(data)
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.date = data['date']
        self.stock_price_history = data['adj_close']
        self.commission_fee = commission_fee
        self.slippage_cost = slippage_cost
        
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        
        self.render_df = pd.DataFrame()
        self.done = False
        self.current_portfolio_value = initial_balance
      
    def reset(self, seed=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.done = False
        self.current_portfolio_value = self.initial_balance
        return self._get_observation(), {}
    
    def step(self, action):
        assert self.action_space.contains(action)
        prev_portfolio_value = self.balance if self.current_step == 0 else self.balance + self.stock_owned * self.stock_price_history[self.current_step - 1]
        current_price = self.stock_price_history[self.current_step]    
        amount = int(self.initial_balance * action[1] / current_price)
    
        if action[0] > 0:  # Buy
            amount = min(int(self.initial_balance * action[1] / current_price), int(self.balance / current_price * (1 + self.commission_fee + self.slippage_cost)))
            if self.balance >= current_price * amount * (1 + self.commission_fee + self.slippage_cost):
                self.stock_owned += amount
                self.balance -= current_price * amount * (1 + self.commission_fee + self.slippage_cost)
        elif action[0] < 0:  # Sell
            amount = min(amount, self.stock_owned)
            if self.stock_owned > 0:
                self.stock_owned -= amount
                self.balance += current_price * amount * (1 - self.commission_fee - self.slippage_cost)
        
        current_portfolio_value = self.balance + self.stock_owned * current_price
        excess_return = current_portfolio_value - prev_portfolio_value 
        risk_free_rate = 0.02  # Example risk-free rate
        std_deviation = np.std(self.stock_price_history[:self.current_step + 1])
        sharpe_ratio = (excess_return - risk_free_rate) / std_deviation if std_deviation != 0 else 0
        reward = sharpe_ratio
         
        self.render(action, amount, current_portfolio_value)
        obs = self._get_observation()
        
        self.current_step += 1
        
        if self.current_step == len(self.data['adj_close']):
            done = True
        else:
            done = False
        
        self.done = done

        info = {}  
        return obs, reward, done, False, info
    
    def _get_observation(self):
        return np.array([
            self.balance,
            self.stock_owned,
            self.stock_price_history[self.current_step],
            self.stock_price_history[self.current_step] - self.stock_price_history[self.current_step - 1] if self.current_step > 0 else 0
        ])
    
    def render(self, action, amount, current_portfolio_value, mode=None):
        current_date = self.date[self.current_step]
        today_action = 'buy' if action[0] > 0 else 'sell'
        current_price = self.stock_price_history[self.current_step]
        
        if mode == 'human':
            print(f"Step:{self.current_step}, Date: {current_date}, Market Value: {current_portfolio_value:.2f}, Balance: {self.balance:.2f}, Stock Owned: {self.stock_owned}, Stock Price: {current_price:.2f}, Today Action: {today_action}:{amount}")
        else:
            pass
        dict = {
            'Date': [current_date], 'market_value': [current_portfolio_value], 'balance': [self.balance], 'stock_owned': [self.stock_owned], 'price': [current_price], 'action': [today_action], 'amount': [amount]
        }
        step_df = pd.DataFrame.from_dict(dict)
        self.render_df = pd.concat([self.render_df, step_df], ignore_index=True)

    def render_all(self):
        df = self.render_df.set_index('Date')       
        fig, ax = plt.subplots(figsize=(18, 6)) 
        df.plot(y="market_value", use_index=True, ax=ax, style='--', color='lightgrey') 
        df.plot(y="price", use_index=True, ax=ax, secondary_y=True, color='black')
         
        for idx in df.index.tolist():
            if (df.loc[idx]['action'] == 'buy') & (df.loc[idx]['amount'] > 0):
                plt.plot(
                    idx,
                    df.loc[idx]["price"] - 1,
                    'g^'
                )
                plt.text(idx, df.loc[idx]["price"]- 3, df.loc[idx]['amount'], c='green', fontsize=8, horizontalalignment='center', verticalalignment='center')
            elif (df.loc[idx]['action'] == 'sell') & (df.loc[idx]['amount'] > 0):
                plt.plot(
                    idx,
                    df.loc[idx]["price"] + 1,
                    'rv'
                )
                plt.text(idx, df.loc[idx]["price"] + 3, df.loc[idx]['amount'], c='red', fontsize=8, horizontalalignment='center', verticalalignment='center')
        plt.show()
```

### `add_indicators` Function
This function adds three technical indicators to the DataFrame: 20-day Moving Average (MA20), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).

- **`df['MA20']`**: Adds a 20-day moving average of the adjusted closing prices.
- **`df['RSI']`**: Adds the Relative Strength Index, which measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
- **`df['MACD']`**: Adds the Moving Average Convergence Divergence, which indicates changes in the strength, direction, momentum, and duration of a trend.

### `StockTradingEnv` Class
This class defines a custom environment for stock trading using the Gymnasium library.

#### `__init__` Method
- **`data`**: The historical stock data.
- **`initial_balance`**: The starting balance for the trading agent.
- **`commission_fee`**: The fee charged for each trade.
- **`slippage_cost`**: The cost incurred due to the difference between the expected price of a trade and the actual price.

Variables:
- **`self.data`**: The stock data with added indicators.
- **`self.current_step`**: The current time step in the environment.
- **`self.initial_balance`**: The initial balance of the trading agent.
- **`self.balance`**: The current balance of the trading agent.
- **`self.stock_owned`**: The number of stocks currently owned by the agent.
- **`self.date`**: The dates corresponding to the stock data.
- **`self.stock_price_history`**: The historical adjusted closing prices of the stock.
- **`self.commission_fee`**: The commission fee for each trade.
- **`self.slippage_cost`**: The slippage cost for each trade.
- **`self.action_space`**: The action space, which includes buying, holding, and selling actions.
- **`self.observation_space`**: The observation space, which includes the current balance, stock owned, current stock price, and price change.
- **`self.render_df`**: A DataFrame to store the rendering data.
- **`self.done`**: A flag indicating whether the episode is done.
- **`self.current_portfolio_value`**: The current value of the portfolio.

#### `reset` Method
Resets the environment to its initial state.
- **`self.current_step`**: Resets to 0.
- **`self.balance`**: Resets to the initial balance.
- **`self.stock_owned`**: Resets to 0.
- **`self.done`**: Resets to False.
- **`self.current_portfolio_value`**: Resets to the initial balance.

#### `step` Method
Executes one time step within the environment.
- **`action`**: The action to be taken by the agent.
- **`prev_portfolio_value`**: The portfolio value before taking the action.
- **`current_price`**: The current stock price.
- **`amount`**: The number of stocks to buy or sell.
- **`reward`**: The reward for the action, calculated as the Sharpe ratio.
- **`obs`**: The new observation after taking the action.
- **`done`**: A flag indicating whether the episode is done.
- **`info`**: Additional information.

#### `_get_observation` Method
Returns the current state of the environment.
- **`self.balance`**: The current balance.
- **`self.stock_owned`**: The number of stocks owned.
- **`self.stock_price_history[self.current_step]`**: The current stock price.
- **`self.stock_price_history[self.current_step] - self.stock_price_history[self.current_step - 1]`**: The price change from the previous step.

#### `render` Method
Renders the environment state for visualization.
- **`action`**: The action taken.
- **`amount`**: The number of stocks bought or sold.
- **`current_portfolio_value`**: The current value of the portfolio.
- **`mode`**: The rendering mode (e.g., 'human').

#### `render_all` Method
Plots the entire trading history.
- **`df`**: The DataFrame containing the rendering data.
- **`fig, ax`**: The figure and axis for plotting.
- **`idx`**: The index of the DataFrame.
- **`plt.plot`**: Plots the buy and sell actions on the graph.
- **`plt.text`**: Adds text annotations for the buy and sell amounts.

This code sets up a reinforcement learning environment for stock trading, allowing an agent to learn and make trading decisions based on historical stock data. The environment includes technical indicators, action and observation spaces, and methods for resetting, stepping, and rendering the environment.


### Code Snippet
```python
env = StockTradingEnv(dftrain, initial_balance=100000, commission_fee=0.0001, slippage_cost=0.005)
modelname = "ppo_" + ticker

tr = input("Train the model? (y/n): ")

if tr == 'y':
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=1e-4,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        ),
        verbose=0
    )
    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(os.path.join('models', modelname))
```

### Context of the Project
This code is part of a larger project that uses reinforcement learning to create a stock trading agent. The agent is trained using the Proximal Policy Optimization (PPO) algorithm to make buy, hold, or sell decisions based on historical stock data. The goal is to maximize the agent's portfolio value over time.

### Detailed Explanation

#### Environment Setup
```python
env = StockTradingEnv(dftrain, initial_balance=100000, commission_fee=0.0001, slippage_cost=0.005)
```
- **`env`**: An instance of the `StockTradingEnv` class, which is the custom environment for stock trading.
- **`dftrain`**: The DataFrame containing the training data (historical stock prices).
- **`initial_balance`**: The starting balance for the trading agent, set to 100,000 units of currency.
- **`commission_fee`**: The fee charged for each trade, set to 0.01% of the trade value.
- **`slippage_cost`**: The cost incurred due to the difference between the expected price of a trade and the actual price, set to 0.5% of the trade value.

#### Model Name
```python
modelname = "ppo_" + ticker
```
- **`modelname`**: The name of the model, which includes the prefix "ppo_" followed by the stock ticker symbol. This helps in identifying the model associated with a particular stock.

#### User Input for Training
```python
tr = input("Train the model? (y/n): ")
```
- **`tr`**: A variable that stores the user's input. The user is prompted to decide whether to train the model or not.

#### Model Training
```python
if tr == 'y':
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=1e-4,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        ),
        verbose=0
    )
    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(os.path.join('models', modelname))
```
- **`if tr == 'y':`**: Checks if the user chose to train the model.
- **`model = PPO(...)`**: Initializes the PPO model with the following parameters:
  - **`"MlpPolicy"`**: The policy network architecture, which is a Multi-Layer Perceptron (MLP).
  - **`env`**: The custom stock trading environment.
  - **`learning_rate=1e-4`**: The learning rate for the optimizer.
  - **`batch_size=256`**: The batch size for training.
  - **`n_steps=2048`**: The number of steps to run for each environment per update.
  - **`gamma=0.99`**: The discount factor for future rewards.
  - **`gae_lambda=0.95`**: The lambda parameter for Generalized Advantage Estimation.
  - **`n_epochs=10`**: The number of epochs to train the model.
  - **`ent_coef=0.01`**: The coefficient for the entropy term, which encourages exploration.
  - **`policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])`**: The architecture of the policy and value networks, each with two hidden layers of 128 units.
  - **`verbose=0`**: The verbosity level of the training output.
- **`model.learn(total_timesteps=100_000, progress_bar=True)`**: Trains the model for 100,000 timesteps, displaying a progress bar.
- **`model.save(os.path.join('models', modelname))`**: Saves the trained model to the specified directory with the given model name.

### Summary
- **Environment Setup**: Initializes the custom stock trading environment with historical data, initial balance, commission fee, and slippage cost.
- **Model Name**: Constructs a model name based on the stock ticker.
- **User Input**: Prompts the user to decide whether to train the model.
- **Model Training**: If the user chooses to train, initializes and trains a PPO model with specified hyperparameters, then saves the trained model.

This code is crucial for setting up the reinforcement learning environment, training the model, and saving it for future use. The trained model can then be used to make trading decisions based on the learned policy.
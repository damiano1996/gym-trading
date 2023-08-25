import warnings

import gym
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

from gym_trading.envs.action_space import BudgetAllocationSpace
from gym_trading.envs.data_loader import PandasAssetChartDataLoader
from gym_trading.envs.exchange import BaseExchange
from gym_trading.envs.renderer import PlotRenderer
from gym_trading.envs.rewards import ProfitRewarder

# Hide all warnings
warnings.filterwarnings("ignore")


def get_symbol_history(symbol, n_days):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': str(n_days),
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Extract dates and prices from the API response
    timestamps = data['prices']
    dates = [pd.to_datetime(timestamp, unit='ms') for timestamp, _ in timestamps]
    prices = [price for _, price in timestamps]
    market_caps = [market_cap for _, market_cap in data['market_caps']]
    total_volumes = [total_volume for _, total_volume in data['total_volumes']]

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Market Cap': market_caps,
        'Total Volume': total_volumes
    })


N_DAYS = 360 * 3

SYMBOLS = ['bitcoin', 'ethereum', 'litecoin', 'ripple']
datasets = {}

df = pd.DataFrame()

for symbol in SYMBOLS:
    df = get_symbol_history(symbol, N_DAYS)

    plt.plot(df['Date'], df['Price'] / np.max(df['Price']), label=symbol)
    datasets[symbol] = df

plt.title('Normalized prices')
plt.legend()
plt.show()


def train_valid_test_df(df, split_rate=0.2):
    train_df, test_df = train_test_split(df, test_size=split_rate, shuffle=False)
    train_df, valid_df = train_test_split(train_df, test_size=split_rate, shuffle=False)
    return train_df, valid_df, test_df


split_datasets = {}
for symbol, df in datasets.items():
    split_datasets[symbol] = train_valid_test_df(df)

    plt.plot(split_datasets[symbol][0]['Date'],
             split_datasets[symbol][0]['Price'] / np.max(split_datasets[symbol][0]['Price']),
             label=f'{symbol} - train')
    plt.plot(split_datasets[symbol][1]['Date'],
             split_datasets[symbol][1]['Price'] / np.max(split_datasets[symbol][1]['Price']),
             label=f'{symbol} - validation')
    plt.plot(split_datasets[symbol][2]['Date'],
             split_datasets[symbol][2]['Price'] / np.max(split_datasets[symbol][2]['Price']),
             label=f'{symbol} - test')

plt.title('Split data')
plt.legend()
plt.show()

INIT_LIQUIDITY = 100.0
BUY_FEE = 0.1
SELL_FEE = 0.1

train_data_loader = PandasAssetChartDataLoader(
    datasets={symbol: split[0] for symbol, split in split_datasets.items()},
    timestamp_column_name='Date',
    price_column_name='Price'
)
train_env = gym.make(
    'gym_trading:trading-v0',
    data_loader=train_data_loader,
    exchange=BaseExchange(train_data_loader, init_liquidity=INIT_LIQUIDITY, buy_fee=BUY_FEE, sell_fee=SELL_FEE),
    rewarder=ProfitRewarder(),
    renderer=PlotRenderer(),
)

valid_data_loader = PandasAssetChartDataLoader(
    datasets={symbol: split[1] for symbol, split in split_datasets.items()},
    timestamp_column_name='Date',
    price_column_name='Price'
)
valid_env = gym.make(
    'gym_trading:trading-v0',
    data_loader=valid_data_loader,
    exchange=BaseExchange(valid_data_loader, init_liquidity=INIT_LIQUIDITY, buy_fee=BUY_FEE, sell_fee=SELL_FEE),
    rewarder=ProfitRewarder(),
    renderer=PlotRenderer(),
)

init_data = train_env.reset()[0]
print(f'Original data:\n{init_data}\n({init_data.shape = })')

done = False
while not done:
    # observation, reward, done, _ = train_env.step(BudgetAllocationSpace(len(split_datasets.keys())).sample())
    observation, reward, done, _ = train_env.step(np.ones(len(split_datasets.keys())) / len(split_datasets.keys()))
    print(observation, observation.shape)
train_env.render()


class ObservationPreprocessWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super(ObservationPreprocessWrapper, self).__init__(venv)

    def reset(self):
        obs = self.venv.reset()
        preprocessed_obs = self._preprocess(obs)
        return preprocessed_obs

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        preprocessed_obs = self._preprocess(obs)
        return preprocessed_obs, rew, done, info

    @staticmethod
    def _preprocess(obs):
        return StandardScaler().fit_transform(obs)


model = PPO(
    policy="MlpPolicy",
    env=ObservationPreprocessWrapper(DummyVecEnv([lambda: train_env])),
    verbose=1,
    tensorboard_log='./logs/',
)

eval_callback = EvalCallback(
    eval_env=ObservationPreprocessWrapper(DummyVecEnv([lambda: valid_env])),
    best_model_save_path='./models',
    log_path='./logs/',
    eval_freq=1000,
    deterministic=True,
    render=False
)

callbacks = [eval_callback]

model.learn(
    total_timesteps=25000,
    callback=callbacks,
    progress_bar=True
)

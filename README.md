# Trading Gym
The Trading Gym is a gym environment for simulating and testing trading strategies using historical price data. 
It is built upon the OpenAI Gym framework and provides a customizable environment for developing and evaluating trading algorithms.

Review this [jupyter notebook](examples/example.ipynb) to learn more about how to use the library.

## Features
- Integration with OpenAI Gym: The Trading Gym extends the functionality of OpenAI Gym to provide a trading-specific environment for reinforcement learning and algorithmic trading research.
- Customizable Data Loader: Load historical price data from various sources and formats, such as CSV files, API calls, or databases, using the flexible Data Loader interface.
- Exchange Simulation: Simulate trading actions, order execution, and portfolio management with the Exchange component. It provides an interface to interact with the market and simulate trading decisions.
- Rendering Options: Visualize price data, trading actions, and portfolio performance using different rendering options, such as plotting, logging, or custom renderers.
- Reward Calculation: Define custom reward functions to evaluate the performance of trading strategies based on specific criteria, such as profit and loss, risk-adjusted returns, or other metrics.
- Observation Window: Define the number of previous price points to include in the observation space, allowing agents to capture historical trends and patterns.

## Installation
To install the Trading Gym, follow these steps:

### Clone the repository:

```commandline
git clone https://github.com/damiano1996/gym-trading.git
```

### Navigate to the cloned directory:

```commandline
cd trading-gym
```

### Create a virtual environment (optional but recommended):

```commandline
python3 -m venv venv
```

### Activate the virtual environment:

#### For Windows:

```commandline
venv\Scripts\activate
```

#### For Unix or Linux:

```commandline
source venv/bin/activate
```

### Install the required dependencies:

```commandline
pip install -r requirements.txt
```

### Start using the Trading Gym in your projects!

## Usage
The following code snippet demonstrates a basic usage example of the Trading Gym:

```python
# Import necessary packages
import gym
import numpy as np

from gym_trading.envs.data_loader import ListAssetChartDataLoader
from gym_trading.envs.exchange import BaseExchange
from gym_trading.envs.renderer import PlotRenderer
from gym_trading.envs.rewards import ProfitRewarder

# Create the Trading Gym environment
env = gym.make(
    'gym_trading:trading-v0',
    data_loader=ListAssetChartDataLoader(...),
    exchange=BaseExchange(...),
    rewarder=ProfitRewarder(),
    renderer=PlotRenderer(),
    observation_window_size=10
)

# Reset the environment and obtain the initial observation
observation = env.reset()[0]

# Simulate a trading session
done = False
while not done:
    # Choose a random action
    action = np.random.randint(0, env.action_space.n)

    # Perform the action and receive the next observation and reward
    observation, reward, done, truncated, _ = env.step(action)

    # Custom logic and analysis can be performed here

# Render the final state of the environment
env.render()
```

For more details on the Trading Gym API review the [jupyter notebook example](examples/example.ipynb).

## License
The Trading Gym is released under the MIT License. Feel free to use, modify, and distribute the code as permitted by the license.

## Acknowledgements
The Trading Gym was inspired by the OpenAI Gym and aims to provide a specialized environment for trading research and algorithmic trading development.

## Happy trading!
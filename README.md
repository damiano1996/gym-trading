# Trading Gym: A Reinforcement Learning Environment for Trading and Budget Allocation

The Trading Gym is a versatile Python library that offers a comprehensive environment for simulating and testing trading strategies, as well as performing budget allocation across a portfolio of assets. Built on the foundation of the OpenAI Gym framework, it provides researchers and traders with a powerful toolkit to develop and evaluate trading algorithms.

## Key Features

### Integration with OpenAI Gym
- The Trading Gym seamlessly integrates with OpenAI Gym, enhancing its capabilities to cater specifically to reinforcement learning and algorithmic trading research.

### Customizable Data Loader
- Load historical price data from a variety of sources and formats, including CSV files, API calls, or databases, using the flexible Data Loader interface. This feature enables you to work with real-world data or synthetic data tailored to your needs.

### Exchange Simulation
- Simulate trading actions, order execution, and portfolio management using the Exchange component. This interface allows you to interact with the market, execute trades, and evaluate trading decisions within a controlled environment.

### Rendering Options
- Visualize price data, trading actions, and portfolio performance through diverse rendering options. You can choose from various visualization methods, including plotting, logging, or even implement custom renderers to suit your visualization requirements.

### Reward Calculation
- Define and implement custom reward functions to evaluate the performance of your trading strategies. You can tailor these functions to measure various criteria, such as profit and loss, risk-adjusted returns, or other specific metrics relevant to your trading objectives.

### Budget Allocation
- In addition to trading, the Trading Gym extends its utility to budget allocation. It allows you to allocate funds across a set of assets, making it suitable for a broader range of financial optimization tasks beyond pure trading strategies.

Whether you're a researcher exploring reinforcement learning in finance or a trader looking to develop and test your trading strategies, the Trading Gym offers a versatile and adaptable environment to meet your needs. To dive deeper into its functionalities and see practical examples, refer to the [jupyter notebook](examples/example.ipynb) provided in the repository.

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
from gym_trading.envs.renderer import PyGamePlotRenderer
from gym_trading.envs.rewards import ProfitRewarder

# Create the Trading Gym environment
env = gym.make(
    'gym_trading:trading-v0',
    data_loader=ListAssetChartDataLoader(...),
    exchange=BaseExchange(...),
    rewarder=ProfitRewarder(),
    renderer=PyGamePlotRenderer(),
    final_report_plot=False
)

# Reset the environment and obtain the initial observation
observation = env.reset()[0]

# Simulate a trading session
done = False
while not done:
    # Choose a random action
    action = env.action_space.sample()  # Sample a random action from the action space

    # Perform the action and receive the next observation and reward
    observation, reward, done, truncated, _ = env.step(action)

    # Custom logic and analysis can be performed here

# Render the final state of the environment
env.render()

# Close the environment
env.close()
```

For more details on the Trading Gym API review the [jupyter notebook example](examples/example.ipynb).

## License
The Trading Gym is released under the MIT License. Feel free to use, modify, and distribute the code as permitted by the license.

## Acknowledgements
The Trading Gym was inspired by the OpenAI Gym and aims to provide a specialized environment for trading research and algorithmic trading development.

## Happy trading!
from stable_baselines3.common.type_aliases import GymEnv
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

# These online portfolio implemntations are based on the alorithms described in:
# Online Portfolio Selection: A Survey by Bin Li, Steven C. H. Hoi
# https://arxiv.org/abs/1212.2129

# Implementation details are also based on examples found here:
# https://github.com/Marigold/universal-portfolios

class CRPModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None # If none, default to uniform weights
            ) -> None:
        
        # WARNING: target weigghts are reordered in the portfolio opimization environment alphabetically
        # Be careful that tartget_weights are in the correct order when passed in here..
        
        # Super simple algorithm, we only need the environment
        assert env is not None 
        self.env = env

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, (defualt to uniform)
        # Uniform base case
        # Note these are the first weight reprsents the cash account, which should always be 0
        self.target_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset
        # Append 0 to the beginning, for an empty cash account
        self.target_weights = np.insert(self.target_weights, 0, 0)
        
        if target_weights is not None:
            # Assert that the portfolio length matches (index 0 represents cash account)
            assert len(target_weights) == self.portfolio_length
            self.target_weights = np.array(target_weights)

    def train(self) -> None:
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'train' on a benchmark model, use predict instead. These models are deterministic.")

    def learn(
        self
    ):
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'learn' on a benchmark model, use predict instead. These models are deterministic.")

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # This is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # This comes from the policies class in stable baselines.
        # Use this to validate the environment.
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        # We always just return the target CRP weights
        actions = self.target_weights.reshape(1, self.portfolio_length)

        # The state doesnt matter here
        return actions, None

class BAHModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None # If none, default to uniform weights
            ) -> None:
        
        # Super simple algorithm, we only need the environment

        assert env is not None 
        self.env = env

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, (defualt to uniform)
        # Uniform base case
        # Note these are the first weight reprsents the cash account, which should always be 0
        self.target_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset
        # Append 0 to the beginning, for an empty cash account
        self.target_weights = np.insert(self.target_weights, 0, 0)
        
        if target_weights is not None:
            # Assert that the portfolio length matches (index 0 represents cash account)
            assert len(target_weights) == self.portfolio_length
            self.target_weights = np.array(target_weights)

    def train(self) -> None:
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'train' on a benchmark model, use predict instead. These models are deterministic.")

    def learn(
        self
    ):
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'learn' on a benchmark model, use predict instead. These models are deterministic.")

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # This is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # This comes from the policies class in stable baselines.
        # Use this to validate the environment.
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        # For BAH we just use whatever weights are already in the environment.

        # For the first step we need to do the initial buy, after that we just let the portfolio run

        # If we are on first time-step, buy the portfolio
        actions = self.target_weights.reshape(1, self.portfolio_length)

        # else 
        if len(self.env._actions_memory) > 1:
            # Use the last portfolio as the new action (keep it the same)
            actions = self.env._final_weights[-1].reshape(1, self.portfolio_length)

        # The state doesnt matter here
        return actions, None
    
class BCRPModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            ) -> None:
        
        # Super simple algorithm, we only need the environment

        assert env is not None 
        self.env = env

        # This portoflio cheats by pulling the full price range ange getting the best portfolio weights in hindsight
        self._full_hindsight_prices = self.env._df

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, for BCRP we use hinesight to get the best possible weights over the time range
        # Here we will cheat and calculate what the best weights would have been
        # This is obviously a benchmark metric and does not work in reality (because we can't see into the future)
        
        # Pivot the DataFrame 
        pivoted_df = self._full_hindsight_prices.pivot(index='date', columns='tic', values='close') 
        # Calculate price ratios 
        price_ratios = pivoted_df / pivoted_df.iloc[0]
        # Get the magic weights
        self.target_weights = np.array(optimize_log_returns(price_ratios))
        # Assume no cash
        self.target_weights = np.insert(self.target_weights, 0, 0)

    def train(self) -> None:
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'train' on a benchmark model, use predict instead. These models are deterministic.")

    def learn(
        self
    ):
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'learn' on a benchmark model, use predict instead. These models are deterministic.")

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # This is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # This comes from the policies class in stable baselines.
        # Use this to validate the environment.
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        # We always just return the target BCRP weights
        actions = self.target_weights.reshape(1, self.portfolio_length)

        # The state doesnt matter here
        return actions, None
    
class SCRPModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            price_history: Optional[pd.DataFrame] = None
            ) -> None:
        
        # Super simple algorithm, we only need the environment

        assert env is not None 
        self.env = env

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]
        
        self.price_history = pd.DataFrame()

        # If available, set additional price history
        if price_history is not None:
            self.price_history = price_history

        # Start with uniform portfolio weights
        self.current_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset
        # Append 0 to the beginning, for an empty cash account
        self.current_weights = np.insert(self.current_weights, 0, 0)

    def train(self) -> None:
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'train' on a benchmark model, use predict instead. These models are deterministic.")

    def learn(
        self
    ):
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'learn' on a benchmark model, use predict instead. These models are deterministic.")

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # This is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # This comes from the policies class in stable baselines.
        # Use this to validate the environment.
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        # Reshape the array to remove single dimensions 
        reshaped_array = observation.reshape(len(self.env._features), self.portfolio_length - 1) 

        # This assumes the close prices are in the environment.
        # This will break if close prices are not first in the environment
        # TODO fix this
        prices = reshaped_array[0].tolist()

        new_row = pd.DataFrame([prices])

        # Add to the price history
        self.price_history = pd.concat([self.price_history, new_row], ignore_index=True)

        # Normalize the prices
        r = {}
        for name, s in self.price_history.items():
            init_val = s.loc[s.first_valid_index()]
            r[name] = s / init_val
        price_history = pd.DataFrame(r)

        if len(price_history) <= 1:
            action_weights = self.current_weights
            actions = action_weights.reshape(1, self.portfolio_length)
            return actions, None
        
        # Find the optimal portfolio over the window price history
        self.current_weights = np.array(optimize_log_returns(price_history))

        assert np.isclose(self.current_weights.sum(), 1), "The array does not sum up to one."

        # Use the last portfolio as the new action (keep it the same)
        action_weights = np.insert(self.current_weights, 0, 0)
        actions = action_weights.reshape(1, self.portfolio_length)

        return actions, None

class OLMARModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None, # If none, default to uniform weights
            window=5, 
            eps=10,
            ) -> None:
        
        # Super simple algorithm, we only need the environment

        assert env is not None 
        self.env = env

        self.window = window
        self.eps = eps

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, (defualt to uniform)
        # Uniform base case
        # Note these are the first weight reprsents the cash account, which should always be 0
        self.current_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset

        # For OLMAR start with uniform and then adjust based on moving averages
        self.price_history = pd.DataFrame()

    def train(self) -> None:
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'train' on a benchmark model, use predict instead. These models are deterministic.")

    def learn(
        self
    ):
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'learn' on a benchmark model, use predict instead. These models are deterministic.")

    def get_SMA(self, window_history):
        """Predict next price relative using SMA."""
        return window_history.mean() / window_history.iloc[-1, :]
        
    def update_weights(self, weights, new_price_prediction):
        """Update portfolio weights to satisfy constraint weights * x >= eps
        and minimize distance to previous weights."""
        price_prediction_mean = np.mean(new_price_prediction)
        excess_return = new_price_prediction - price_prediction_mean
        denominator = (excess_return * excess_return).sum()
        if denominator != 0:
            lam = max(0.0, (self.eps - np.dot(weights, new_price_prediction)) / denominator)
        else:
            lam = 0

        # update portfolio
        weights = weights + lam * (excess_return)

        # project it onto simplex
        return simplex_projection(weights)

    def predict(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False, # This is always determininistic
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # This comes from the policies class in stable baselines.
        # Use this to validate the environment.
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        # Reshape the array to remove single dimensions 
        reshaped_array = observation.reshape(len(self.env._features), self.portfolio_length - 1) 

        # This assumes the close prices are in the environment.
        # This will break if close prices are not first in the environment
        # TODO fix this
        prices = reshaped_array[0].tolist()

        new_row = pd.DataFrame([prices])

        # Add to the price history
        self.price_history = pd.concat([self.price_history, new_row], ignore_index=True)
        old_weights = self.current_weights

        # Normalize the prices
        normals = {}
        for col, close_prices in self.price_history.items():
            init_val = close_prices.loc[close_prices.first_valid_index()]
            normals[col] = close_prices / init_val
        price_relatives = pd.DataFrame(normals)

        # Window is too short, return the starting weights
        if len(price_relatives) < self.window + 1:
            self.price_prediction = price_relatives.iloc[-1]
        else:
            window_history = price_relatives.iloc[-self.window :]
            self.price_prediction = self.get_SMA(window_history)
            
        new_weights = self.update_weights(old_weights, self.price_prediction)

        self.current_weights = new_weights

        assert np.isclose(self.current_weights.sum(), 1), "The array does not sum up to one."

        # Use the last portfolio as the new action (keep it the same)
        action_weights = np.insert(new_weights, 0, 0)
        actions = action_weights.reshape(1, self.portfolio_length)

        return actions, None


class RMRModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None, # If none, default to uniform weights
            window=5, 
            eps=10,
            tau=0.001 # L1 Normilization parameter
            ) -> None:
        
        # Super simple algorithm, we only need the environment

        assert env is not None 
        self.env = env

        self.window = window
        self.eps = eps
        self.tau = tau

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, (defualt to uniform)
        # Uniform base case
        # Note these are the first weight reprsents the cash account, which should always be 0
        self.current_weights = np.ones(self.portfolio_length-1) / (self.portfolio_length-1)  # target weights for each asset

        # For RMR start with uniform and then adjust based on moving averages
        self.price_history = pd.DataFrame()

    def train(self) -> None:
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'train' on a benchmark model, use predict instead. These models are deterministic.")

    def learn(
        self
    ):
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'learn' on a benchmark model, use predict instead. These models are deterministic.")

    def get_price_prediction(self, window_history):
        """Predict next price relative using L1 norm."""
        mean_prices = window_history.mean()
        prev_mean_prices = None
        adjust_prices = False
        while prev_mean_prices is None or adjust_prices:
            prev_mean_prices = mean_prices
            norm = calculate_l1_norm(window_history - mean_prices)
            mean_prices = window_history.div(norm, axis=0).sum() / (1.0 / norm).sum()
            adjust_prices = calculate_l1_norm(mean_prices - prev_mean_prices) / calculate_l1_norm(prev_mean_prices) > self.tau
        return mean_prices
        
    def update_weights(self, weights, new_price_prediction):
        """Update portfolio weights to satisfy constraint weights * x >= eps
        and minimize distance to previous weights."""
        price_prediction_mean = np.mean(new_price_prediction)
        excess_return = new_price_prediction - price_prediction_mean
        denominator = (excess_return * excess_return).sum()
        if denominator != 0:
            lam = max(0.0, (self.eps - np.dot(weights, new_price_prediction)) / denominator)
        else:
            lam = 0

        # update portfolio
        weights = weights + lam * (excess_return)

        # project it onto simplex
        return simplex_projection(weights)

    def predict(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # Much of this comes from the policies class in stable baselines
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        # Reshape the array to remove single dimensions 
        reshaped_array = observation.reshape(len(self.env._features), self.portfolio_length - 1) 

        # This assumes the close prices are in the environment.
        # This will break if close prices are not first in the environment
        # TODO fix this
        prices = reshaped_array[0].tolist()

        new_row = pd.DataFrame([prices])

        # Add to the price history
        self.price_history = pd.concat([self.price_history, new_row], ignore_index=True)
        old_weights = self.current_weights

        # Normalize the prices
        normals = {}
        for col, close_prices in self.price_history.items():
            init_val = close_prices.loc[close_prices.first_valid_index()]
            normals[col] = close_prices / init_val
        price_relatives = pd.DataFrame(normals)

        current_prices = price_relatives.iloc[-1]

        # Window is too short, just use last price
        if len(price_relatives) < self.window + 1:
            self.price_prediction = current_prices
        else:
            window_history = price_relatives.iloc[-self.window :]
            self.price_prediction = self.get_price_prediction(window_history)
            
        new_weights = self.update_weights(old_weights, self.price_prediction)

        self.current_weights = new_weights

        assert np.isclose(self.current_weights.sum(), 1), "The array does not sum up to one."

        # Use the last portfolio as the new action (keep it the same)
        action_weights = np.insert(new_weights, 0, 0)
        actions = action_weights.reshape(1, self.portfolio_length)

        return actions, None
    
class BNNModel:
    def __init__(
            self, 
            env: Union[GymEnv, str],
            policy: Any, # Policy doesnt matter here
            device: str, # device doesnt matter here
            policy_kwargs: Optional[Dict[str, Any]] = None, # policy_kwargs doesnt matter here
            target_weights: List[float] = None, # If none, default to uniform weights
            window=5, # Sequence length
            neighbors=10 # Number of neighbors
            ) -> None:
        
        # Super simple algorithm, we only need the environment

        assert env is not None 
        self.env = env

        self.window = window
        self.neighbors = neighbors

        self.min_history= window + neighbors - 1

        # Pull out the actions space dimensions for the portfolio
        self.action_space_shape = self.env.action_space.shape
        self.portfolio_length = self.action_space_shape[0]

        # Calculate the inital weights, here we set all weights to zero to start
        # Then we put all value in the cash account

        # Note these are the first weight reprsents the cash account, which should always be 0
        self.current_weights = np.zeros(self.portfolio_length-1)   # target weights for each asset

        self.price_history = pd.DataFrame()

    def find_neighbors(self, price_history):
        """
        Based on the implementation here: 
        https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bnn.py
        """
        # find the distance from the last row to every other row in the window
        distances = price_history * 0
        for i in range(1, self.window + 1):
            lagged_prices = price_history.shift(i - 1)
            last_prices = price_history.iloc[-i]
            distances += (lagged_prices - last_prices) ** 2

        # Calculate the distance from every time-step to the last point (excluding the last time-step)
        distances = distances.sum(1).iloc[:-1]

        # Drop the zero rows from sorting.
        distances = distances[distances != 0]

        # sort the list and return the nearest (minimum distances)
        distances = distances.sort_values()
        return distances.index[:self.neighbors]


    def train(self) -> None:
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'train' on a benchmark model, use predict instead. These models are deterministic.")

    def learn(
        self
    ):
        # This model is derministic and doesnt learn anything, it only predicts
        raise NotImplementedError("Can't use 'learn' on a benchmark model, use predict instead. These models are deterministic.")

    def predict(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # This comes from the policies class in stable baselines.
        # Use this to validate the environment.
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        # Reshape the array to remove single dimensions 
        reshaped_array = observation.reshape(len(self.env._features), self.portfolio_length - 1) 

        # This assumes the close prices are in the environment.
        # This will break if close prices are not first in the environment
        # TODO fix this
        prices = reshaped_array[0].tolist()

        new_row = pd.DataFrame([prices])

        # Add to the price history
        self.price_history = pd.concat([self.price_history, new_row], ignore_index=True)

        # Normalize the prices
        r = {}
        for name, s in self.price_history.items():
            init_val = s.loc[s.first_valid_index()]
            r[name] = s / init_val
        price_history = pd.DataFrame(r)

        # Window is too short, use cash only
        if len(price_history) < self.min_history + 1:
            weights = self.current_weights
            action_weights = np.insert(weights, 0, 1)
            actions = action_weights.reshape(1, self.portfolio_length)
            return actions, None
        
        neighbor_indexes = self.find_neighbors(price_history)

        neighbor_history = price_history.iloc[[price_history.index.get_loc(i) + 1 for i in neighbor_indexes]]
        
        # Find the optimal portfolio over the nearest neighbor price history
        self.current_weights = np.array(optimize_log_returns(neighbor_history))

        assert np.isclose(self.current_weights.sum(), 1), "The array does not sum up to one."

        # Use the last portfolio as the new action (keep it the same)
        action_weights = np.insert(self.current_weights, 0, 0)
        actions = action_weights.reshape(1, self.portfolio_length)

        return actions, None

# Found this here:  
# https://github.com/Marigold/universal-portfolios/blob/master/universal/tools.py
def calculate_l1_norm(prices):
    # Determine the axis along which to calculate the norm
    if isinstance(prices, pd.Series):
        axis = 0
    else:
        axis = 1
    
    # Calculate the L1 norm
    return np.abs(prices).sum(axis=axis)


# Found this here:  
# https://github.com/Marigold/universal-portfolios/blob/master/universal/tools.py
def simplex_projection(weights):
    """Projection of weights onto simplex."""
    weight_length = len(weights)
    found = False

    sorted_weights = sorted(weights, reverse=True)
    running_sum = 0.0

    for weight_i in range(weight_length - 1):
        running_sum = running_sum + sorted_weights[weight_i]
        threshhold_max = (running_sum - 1) / (weight_i + 1)
        if threshhold_max >= sorted_weights[weight_i + 1]:
            found = True
            break

    if not found:
        threshhold_max = (running_sum + sorted_weights[weight_length - 1] - 1) / weight_length

    return np.maximum(weights - threshhold_max, 0.0)   


import scipy.optimize as optimize
# Found this here:  
# https://github.com/Marigold/universal-portfolios/blob/master/universal/tools.py
def optimize_log_returns(
    prices
):
    assert prices.notnull().all().all()

    x_0 = np.ones(prices.shape[1]) / float(prices.shape[1])
    
    # Using price returns here.
    objective = lambda b: -np.sum(np.log(np.maximum(np.dot(prices - 1, b) + 1, 0.0001)))

    cons = ({"type": "eq", "fun": lambda b: 1 - sum(b)},)

    # problem optimization
    res = optimize.minimize(
        objective,
        x_0,
        bounds=[(0.0, 1.0)] * len(x_0),
        constraints=cons,
        method="slsqp"
    )

    if res.success:
        return res.x
    raise ValueError("Could not find an optimal value using the BCRP algorithm.")
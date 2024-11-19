from stable_baselines3 import PPO, A2C

from .online import CRPModel, BAHModel, BCRPModel, OLMARModel, RMRModel, BNNModel

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback


MODELS = {
    "ppo": PPO, 
    "a2c": A2C, 
    "crp": CRPModel, 
    "bah": BAHModel, 
    "bcrp": BCRPModel, 
    "olmar": OLMARModel,
    "rmr": RMRModel,
    "bnn": BNNModel
}

class DRLStableAgent:
    """Implementation for DRL algorithms for portfolio optimization.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy is updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        env: Gym environment class.
    """

    def __init__(self, env):
        """Agent initialization.

        Args:
            env: Gym environment to be used in training.
        """
        self.env = env

    def get_model(
        self, model_name, device="cpu", model_kwargs=None, policy_kwargs=None
    ):
        """Setups DRL model.

        Args:
            model_name: Name of the model according to MODELS list.
            device: Device used to instantiate neural networks.
            model_kwargs: Arguments to be passed to model class.
            policy_kwargs: Arguments to be passed to policy class.

        Note:
            model_kwargs and policy_kwargs are dictionaries. The keys must be strings
            with the same names as the class arguments. Example for model_kwargs::

            { "lr": 0.01, "policy": EIIE }

        Returns:
            An instance of the model.
        """
        if model_name not in MODELS:
            raise NotImplementedError("The model requested was not implemented.")

        model = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # add device settings
        model_kwargs["device"] = device
        #policy_kwargs["device"] = device

        # add policy_kwargs inside model_kwargs
        model_kwargs["policy_kwargs"] = policy_kwargs

        # Default to use the MlpPolicy
        return model(env=self.env, policy="MlpPolicy", **model_kwargs)

    @staticmethod
    def train_model(model, env, tb_log_name=None, episodes=1):
        """Trains portfolio optimization model.

        Args:
            model: Instance of the model.
            episoded: Number of episodes.

        Returns:
            An instance of the trained model.
        """
        max_steps = len(env._df['date'].unique())

        print("Max number of time steps in an episode: ", max_steps)

        checkpoint_callback = CheckpointCallback(
            save_freq=1000000,
            save_path="./results/",
            name_prefix="model_checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        model.learn(
            total_timesteps = max_steps * episodes,
            callback=checkpoint_callback,
            tb_log_name=tb_log_name,
            progress_bar=True
        )
        return model

    @staticmethod
    def DRL_prediction(model, env, deterministic=True, verbose=False):
        """make a prediction and get results"""
        test_env, test_obs = env.get_sb_env()
        
        test_env.reset()
        max_steps = len(env._df['date'].unique())

        validation_assets = None
        validation_dates = None

        for i in range(max_steps):
            action, _states = model.predict(test_obs, deterministic=deterministic)

            if(verbose):
                print("Step: ", str(i))
                print("Observations: ")
                print(test_obs)
                print("Actions: ")
                print(action)

            # Pull out the latest assets and dates
            validation_assets = env._asset_memory["final"]
            validation_dates = env._date_memory

            test_obs, rewards, dones, info = test_env.step(action)

            if dones[0]:
                print("hit end!")
                break
        
        return validation_assets, validation_dates
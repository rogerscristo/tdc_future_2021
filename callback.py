# Modified version of https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from IPython.display import clear_output
from utils import write_yaml, read_yaml

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param total_timesteps: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, total_timesteps: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_models')
        self.best_mean_reward = {"mean": .0,
                                 "std": .0}
        self.total_timesteps = total_timesteps
        self.metadata = {}
        self.models_saved_count = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
         
        
    def _print_log_jupyter(self):
        clear_output(wait=True)
        title_line = "\U0000300B"
        output = f"""
        {title_line*5} SEED \U0001F3B2 {self.model.seed}
        {title_line*5} Timesteps \U0001F3C3 {self.num_timesteps}/{self.total_timesteps}
        {title_line*5} Best reward \U0001F947 {self.best_mean_reward.get("mean"):.2f} \U000000B1 {self.best_mean_reward.get("std"):.2f}
        """
        print(output)
        

    def _on_step(self) -> bool:
        if self.verbose > 0:
            self._print_log_jupyter()
            
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                std_reward = np.std(y[-100:])
                
                if mean_reward > self.best_mean_reward.get("mean"):
                    best_model_path = os.path.join(self.save_path, f"{self.num_timesteps}steps_{int(mean_reward)}rew")
                    self.best_mean_reward.update({"mean": mean_reward})
                    self.best_mean_reward.update({"std": std_reward})
                    self.model.save(best_model_path)
                    self.metadata[self.models_saved_count] = {"steps": self.num_timesteps, 
                                                              "mean_reward": int(mean_reward),
                                                              "std_reward": int(std_reward), 
                                                              "relative_path": best_model_path}
                    self.models_saved_count += 1

        return True
    
    def _on_training_end(self) -> None:
        write_yaml(self.metadata, self.save_path)
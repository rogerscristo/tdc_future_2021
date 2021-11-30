import os, yaml, imageio, pybullet_envs
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from IPython.display import display, HTML

def load_eval_env(stats_path, env_id, seed, n_envs, wrapper_class):
    env = make_vec_env(env_id=env_id,
                       seed=seed,
                       n_envs=n_envs,
                       wrapper_class=wrapper_class)
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    return env

def eval_model(env, model):
    images = []
    obs = env.reset()
    img = env.render(mode='rgb_array')
    
    for i in range(200):
        action, _ = model.predict(obs)
        obs, _, _ ,_ = env.step(action)
        img = env.render(mode='rgb_array')
        images.append(img)
    
    return images

def images2gif(images, filename, fps=29):
    imageio.mimsave(f"{filename}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=fps)
    
def write_yaml(data, path):
    """ A function to write YAML file"""
    with open(os.path.join(path, "metadata.yml"), 'a') as file:
        yaml.dump(data, file, default_flow_style=False)
        
def read_yaml(path):
    """ A function to read YAML file"""
    with open(path) as f:
        yaml_file = yaml.safe_load(f)
 
    return yaml_file

def eval2gif(best_models, eval_env, algo, log_dir) -> None:
    metadata = {}
    gifs_dir = os.path.join(log_dir, "result_animations")
    os.makedirs(gifs_dir, exist_ok=True)
    
    for key, value in best_models.items():
        model = algo.load(value["relative_path"])
        images = eval_model(eval_env, model)

        gif_path = f"{value['steps']}_steps_{value['mean_reward']}_rew"
        metadata[key] = {"steps": value["steps"], 
                         "mean_reward": value["mean_reward"], 
                         "std_reward": value["std_reward"], 
                         "relative_path": os.path.join(gifs_dir, f"{gif_path}.gif")}
        images2gif(images, os.path.join(gifs_dir, gif_path))
    
    write_yaml(metadata, gifs_dir)
    
def get_gifs_paths(metadata_path) -> list:
    gifs_metadata = read_yaml(metadata_path)
    
    return [value["relative_path"] for key, value in gifs_metadata.items()]

def get_experiments_animations(metadata_path) -> str:
    def make_html(image_path, label) -> str:
        return '<div style="display: inline-block; text-align: center;">{}<img src="{}" style="display:block;margin:1px"/></div>'.format(label, image_path)
    
    metadata = read_yaml(metadata_path)
    
    gifs_html = [make_html(os.path.join(os.path.abspath(os.getcwd()), value["relative_path"]), f"steps: {value['steps']} | mean reward {value['mean_reward']} +- {value['std_reward']}") for key, value in metadata.items()]
    return ''.join(gifs_html)
        
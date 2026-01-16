# train_advanced.py - Gelişmiş RL Eğitimi

import os
import sys
import numpy as np
from datetime import datetime

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn

from environment_advanced import AdvancedCellEnv, CurriculumCellEnv

# GPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DetailedLoggingCallback(BaseCallback):
    """Detaylı eğitim logları"""
    
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        
    def _on_step(self) -> bool:
        # Episode bitti mi kontrol et
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    ep_reward = info.get('episode', {}).get('r', 0)
                    ep_length = info.get('episode', {}).get('l', 0)
                    
                    if ep_reward != 0:
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
        
        # Periyodik log
        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            
            print(f"\n{'='*60}")
            print(f"Timesteps: {self.num_timesteps}")
            print(f"Mean Reward (last 100): {mean_reward:.2f}")
            print(f"Mean Length (last 100): {mean_length:.1f}")
            print(f"Best Reward: {self.best_reward:.2f}")
            print(f"{'='*60}\n")
            
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save("models/best_cell_model")
                print("💾 New best model saved!")
        
        return True


class CustomNetworkCallback(BaseCallback):
    """Ağ istatistiklerini logla"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 == 0:
            # Policy network gradyanlarını kontrol et
            policy = self.model.policy
            total_grad_norm = 0
            for param in policy.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Gradient norm: {total_grad_norm:.4f}")
        
        return True


def make_env(rank: int, seed: int = 0, curriculum: bool = False):
    """Paralel ortam oluşturucu"""
    def _init():
        if curriculum:
            env = CurriculumCellEnv({'max_steps': 2000})
        else:
            env = AdvancedCellEnv({'max_steps': 2000})
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    return _init


def train_sac(total_timesteps: int = 2_000_000, n_envs: int = 4):
    """
    SAC (Soft Actor-Critic) ile eğitim
    Continuous action space için en iyi algoritmalardan biri
    """
    print("\n" + "="*60)
    print("🧬 SAC Training Starting")
    print("="*60 + "\n")
    
    # Ortam oluştur
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, curriculum=True) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, curriculum=True)])
    
    # Eval ortamı
    eval_env = DummyVecEnv([make_env(0)])
    
    # Hyperparametreler
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],  # Policy network
            qf=[512, 512, 256]   # Q-function network
        ),
        activation_fn=nn.ReLU
    )
    
    # Model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',  # Otomatik entropy ayarı
        target_entropy='auto',
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log="./logs/sac_cell"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    logging_callback = DetailedLoggingCallback(log_freq=5000)
    
    # Eğit
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, logging_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
    
    # Kaydet
    model.save("models/sac_cell_final")
    print("\n✅ Training complete! Model saved to models/sac_cell_final")
    
    return model


def train_td3(total_timesteps: int = 2_000_000, n_envs: int = 4):
    """
    TD3 (Twin Delayed DDPG) ile eğitim
    Daha stabil öğrenme
    """
    print("\n" + "="*60)
    print("🧬 TD3 Training Starting")
    print("="*60 + "\n")
    
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(0)])
    
    # Action noise (exploration için)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    policy_kwargs = dict(
        net_arch=[512, 512, 256],
        activation_fn=nn.ReLU
    )
    
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log="./logs/td3_cell"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000
    )
    
    logging_callback = DetailedLoggingCallback(log_freq=5000)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, logging_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
    
    model.save("models/td3_cell_final")
    print("\n✅ Training complete!")
    
    return model


def train_ppo_advanced(total_timesteps: int = 2_000_000, n_envs: int = 8):
    """
    PPO ile gelişmiş eğitim
    Daha kararlı ama yavaş
    """
    print("\n" + "="*60)
    print("🧬 Advanced PPO Training Starting")
    print("="*60 + "\n")
    
    env = SubprocVecEnv([make_env(i, curriculum=True) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(0)])
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],
            vf=[512, 512, 256]
        ),
        activation_fn=nn.Tanh,  # PPO için Tanh daha iyi olabilir
        ortho_init=True
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,  # Exploration teşviki
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log="./logs/ppo_cell"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000
    )
    
    logging_callback = DetailedLoggingCallback(log_freq=5000)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, logging_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
    
    model.save("models/ppo_cell_final")
    print("\n✅ Training complete!")
    
    return model


def hyperparameter_search():
    """
    Basit hyperparameter arama
    """
    from itertools import product
    
    learning_rates = [1e-4, 3e-4, 1e-3]
    batch_sizes = [64, 128, 256]
    net_archs = [
        [256, 256],
        [512, 256],
        [512, 512, 256]
    ]
    
    best_reward = -float('inf')
    best_params = None
    
    for lr, bs, arch in product(learning_rates, batch_sizes, net_archs):
        print(f"\nTrying: lr={lr}, batch_size={bs}, arch={arch}")
        
        env = DummyVecEnv([make_env(0)])
        
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=lr,
            batch_size=bs,
            policy_kwargs=dict(net_arch=dict(pi=arch, qf=arch)),
            verbose=0
        )
        
        model.learn(total_timesteps=50000)
        
        # Değerlendir
        eval_env = make_env(99)()
        rewards = []
        for _ in range(10):
            obs = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        
        mean_reward = np.mean(rewards)
        print(f"Mean reward: {mean_reward:.2f}")
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = {'lr': lr, 'batch_size': bs, 'arch': arch}
    
    print(f"\n🏆 Best params: {best_params}")
    print(f"🏆 Best reward: {best_reward:.2f}")
    
    return best_params


def main():
    # Klasörleri oluştur
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "sac":
            train_sac()
        elif command == "td3":
            train_td3()
        elif command == "ppo":
            train_ppo_advanced()
        elif command == "search":
            hyperparameter_search()
        else:
            print("Usage: python train_advanced.py [sac|td3|ppo|search]")
    else:
        # Varsayılan: SAC
        print("No algorithm specified. Using SAC (recommended for continuous actions).")
        train_sac()


if __name__ == "__main__":
    main()
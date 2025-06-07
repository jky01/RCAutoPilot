#!/usr/bin/env python3
import gymnasium as gym # 或者 import gym
import gym_donkeycar
import uuid
import os
from stable_baselines3 import SAC # 您可以選擇 PPO, DDPG, TD3 等其他算法
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

# --- 配置 ---
SIM_EXE_PATH = "" # 留空以使用 pip 安裝的 gym_donkeycar 模擬器
                  # 或填寫您下載的模擬器執行檔路徑，例如:
                  # SIM_EXE_PATH = "/Applications/donkey_sim.app/Contents/MacOS/donkey_sim"
                  # SIM_EXE_PATH = "C:/Users/YourUser/Downloads/DonkeySimWin/donkey_sim.exe"

ENV_NAME = "donkey-generated-track-v0" # 選擇您想訓練的賽道
                                       # 其他選項: "donkey-warehouse-v0", "donkey-avc-sparkfun-v0", etc.

LOG_DIR = "./rl_logs/" # 儲存 TensorBoard logs 和模型的位置
MODEL_SAVE_PATH = os.path.join(LOG_DIR, "sac_donkey_model") # 訓練好的模型儲存名稱
BEST_MODEL_SAVE_PATH = os.path.join(LOG_DIR, "best_model")

TOTAL_TIMESTEPS = 100000 # 總訓練步數，根據需求調整
LEARNING_RATE = 1e-4 # 學習率，SAC 的典型值較小
BUFFER_SIZE = 50000  # SAC 的 Replay Buffer 大小
BATCH_SIZE = 64      # SAC 的 Batch Size

os.makedirs(LOG_DIR, exist_ok=True)

# --- Donkey Gym 環境配置 ---
# 這些配置會傳遞給模擬器
conf = {
    "exe_path": SIM_EXE_PATH if SIM_EXE_PATH else "self", # "self" 表示使用 gym_donkeycar 內建的模擬器
    "host": "127.0.0.1",
    "port": 9091, # 確保這個端口未被佔用
    "body_style": "donkey",
    "body_rgb": (128, 128, 128),
    "car_name": "MyRLCar",
    "font_size": 10,
    "racer_name": "RL Agent",
    "country": "CN",
    "bio": "Learning to drive with RL",
    "guid": str(uuid.uuid4()), # 每次運行生成唯一 ID
    "max_cte": 7.0, # 最大橫向跟蹤誤差，超過此值 episode 可能會結束 (取決於環境)
    "headless": 0, # 0: 顯示模擬器視窗, 1: 無頭模式 (訓練時建議設為1以加速)
    "log_level": 20, # logging.INFO
    # "cam_resolution": (120, 160, 3) # gym_donkeycar 預設是 120x160x3
}

def make_env():
    """
    Utility function for multiprocessed env.
    """
    env = gym.make(ENV_NAME, conf=conf)
    env = Monitor(env, LOG_DIR) # 記錄訓練過程中的獎勵等信息
    return env

if __name__ == "__main__":
    print("Creating Donkey Car Gym environment...")
    # 使用 DummyVecEnv 來包裹環境，這是 stable-baselines3 的要求
    # 如果您想使用多個並行環境進行訓練 (以加速)，可以使用 SubprocVecEnv
    env = DummyVecEnv([make_env])

    # (可選) 正規化觀察空間和獎勵，對於某些 RL 算法可能會有幫助
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # --- 模型定義 ---
    # 使用 SAC (Soft Actor-Critic) 算法，它在連續動作空間中表現良好
    # "CnnPolicy" 表示使用卷積神經網路來處理圖像輸入
    model = SAC("CnnPolicy",
                env,
                verbose=1,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                tensorboard_log=LOG_DIR)

    print("Starting RL training...")
    # 定期儲存模型的 Callback
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=LOG_DIR, name_prefix="rl_model_checkpoint")
    # 評估並儲存最佳模型的 Callback (需要一個單獨的評估環境)
    # eval_env = Monitor(gym.make(ENV_NAME, conf=conf))
    # eval_callback = EvalCallback(eval_env, best_model_save_path=BEST_MODEL_SAVE_PATH, log_path=LOG_DIR, eval_freq=10000, deterministic=True, render=False)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback]) # 加入 eval_callback 如果設定了

    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    # (可選) 關閉環境
    env.close()

    print("To run the trained agent, configure your myconfig.py and use 'python manage.py drive'")
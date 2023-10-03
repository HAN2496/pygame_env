"""
학습 후 테스트하는 코드 예제
1. 카메이커 연동 환경을 불러온다
2. 학습에 사용한 RL 모델(e.g. PPO)에 학습된 웨이트 파일(e.g. model.pkl)을 로드한다.
3. 테스트를 수행한다.
"""

from lowlevel_env2 import LowLevelEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

if __name__ == '__main__':
    road_type = "DLC"
    data_name = 'LowLevel'
    comment = "myEnv2"
    prefix = data_name + comment


    env = CarEnv()
    """
    env.car.carx, env.car.cary, env.car.caryaw = 68, -5.0525, 0
    traj_abs = env.find_lookahead_traj(env.car.carx, env.car.cary, [3 * i for i in range(5)])
    traj_rel = env.to_relative_coordinates(env.car.carx, env.car.cary, env.car.caryaw, traj_abs)
    devdist, devAng = env.calculate_dev()
    print(traj_rel)
    print(f"dst: {devdist}, ang: {devAng}")
    """
    model = SAC("MlpPolicy", env, tensorboard_log=os.path.join(f"tensorboard/dlc"), verbose=1)
    try:
        model.learn(total_timesteps=10000 * 300)
    except KeyboardInterrupt:
        print("Learning interrupted. Will save the model now.")
    finally:
        model.save("models/blevel/model_l.pkl")
        print("Model saved")

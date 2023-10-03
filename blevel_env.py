import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
import pygame
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import interp1d
from lowlevel_env2 import LowLevelEnv
import time

def plot(road, car):
    #plt.figure(figsize=(10, 5))

    # Plot forbidden areas
    plt.plot(*road.forbbiden_area1.exterior.xy, label="Forbidden Area 1", color='red')
    plt.plot(*road.forbbiden_area2.exterior.xy, label="Forbidden Area 2", color='blue')
    plt.plot(*road.road_boundary.exterior.xy, label="ROAD BOUNDARY", color='green')

    # Plot cones
    cones_x = road.cones_arr[:, 0]
    cones_y = road.cones_arr[:, 1]
    plt.scatter(cones_x, cones_y, s=10, color='orange', label="Cones")

    # Plot the car
    car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
    plt.plot(*car_shape.exterior.xy, color='blue', label="Car")
    plt.scatter(car.carx, car.cary, color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.gca().invert_yaxis()
    plt.title('Car, Forbidden Areas and Cones')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

class Road:
    def __init__(self):
        self.road_length = 161
        self.road_width = -15
        self._forbidden_area()
        self.cones_arr = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()

    def _forbidden_area(self):
        vertices1 = [
            (0, -6.442), (62, -6.442), (62, -3.221), (99, -3.221), (99, -6.442),
            (161, -6.442), (161, 0), (0, 0), (0, -6.442)
        ]
        vertices2 = [
            (0, -9.663), (75.5, -9.663), (75.5, -6.442), (86.5, -6.442),
            (86.5, -9.663), (161, -9.663), (161, -12.884), (0, -12.884), (0, -9.663)
        ]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbbiden_line1 = LineString(vertices1[:])
        self.forbbiden_line2 = LineString(vertices2[:])
        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
        ])
        self.cones_boundary = Polygon(
            [(0, -7.0651), (62, -7.0651), (62, -3.4565), (99, -3.4565), (99, -6.5525), (161, -6.5525),
             (161, -9.5525), (86.5, -9.5525), (86.5, -6.2065), (75.5, -6.2065), (75.5, -9.0399), (0, -9.0399)
        ])

    def create_cone_shape(self):
        sections = self.create_DLC_cone()
        cones = []
        for section in sections:
            for i in range(section['num']):  # Each section has 5 pairs
                x_base = section['start'] + section['gap'] * i
                y1 = section['y_offset'] - section['cone_dist'] / 2
                y2 = section['y_offset'] + section['cone_dist'] / 2
                cone1 = Point(x_base, y1).buffer(0.2)  # 반지름이 0.2m인 원 생성
                cone2 = Point(x_base, y2).buffer(0.2)  # 반지름이 0.2m인 원 생성
                cones.extend([cone1, cone2])

        return np.array(cones)

    def create_cone_arr(self):
        sections = self.create_DLC_cone()
        cones = []
        for section in sections:
            for i in range(section['num']):  # Each section has 5 pairs
                x_base = section['start'] + section['gap'] * i
                y1 = section['y_offset'] - section['cone_dist'] / 2
                y2 = section['y_offset'] + section['cone_dist'] / 2
                cones.extend([[x_base, y1], [x_base, y2]])

        return np.array(cones)

    def create_DLC_cone(self):
        sections = [
            {'start': 0, 'gap': 5, 'cone_dist': 1.9748, 'num': 10, 'y_offset': -8.0525},
            {'start': 50, 'gap': 3, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 64.7, 'gap': 2.7, 'cone_dist': 5.4684, 'num': 4, 'y_offset': -6.3057},
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2, 'num': 5, 'y_offset': -4.8315}, #
            {'start': 89, 'gap': 2.5, 'cone_dist': 5.4684, 'num': 4, 'y_offset': -6.3057},
            {'start': 99, 'gap': 3, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 111, 'gap': 5, 'cone_dist': 1.9748, 'num': 20, 'y_offset': -8.0525}
        ]

        return sections
    def is_car_in_forbidden_area(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)

        if car_shape.intersects(self.forbbiden_area1) or car_shape.intersects(self.forbbiden_area2):
            return 1
        else:
            return 0
    def is_car_colliding_with_cones(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
        for cone in self.cones_shape:
            if car_shape.intersects(cone):
                return 1
        return 0

    def is_car_in_road(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
        if not car_shape.intersects(self.road_boundary):
            return 1
        if not self.road_boundary.contains(car_shape):
            return 1
        return 0

    def is_car_in_cone_area(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
        if not car_shape.intersects(self.cones_boundary):
            return 1
        if not self.cones_boundary.contains(car_shape):
            return 1
        return 0

class Car:
    def __init__(self):
        self.length = 4.3
        self.width = 1.568
        self.carx = 3
        self.cary = -8.0525
        self.caryaw = 0
        self.carv = 13.8889

    def reset_car(self):
        self.carx = 3
        self.cary = -8.0525
        self.caryaw = 0

    def move_car(self, angle):
        angle = angle[0]
        self.caryaw += angle * 0.01
        self.carx += np.cos(self.caryaw) * self.carv * 0.01
        self.cary += np.sin(self.caryaw) * self.carv * 0.01

    def shape_car(self, carx, cary, caryaw):
        half_length = self.length / 2.0
        half_width = self.width / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        car_shape = Polygon(corners)
        car_shape = affinity.rotate(car_shape, caryaw, origin='center', use_radians=False)
        car_shape = affinity.translate(car_shape, carx, cary)

        return car_shape

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        self.reset_num = 0
        self.time = 0
        self.test_num = 0

        self.car = Car()
        self.road = Road()

        env_action_num = 3
        env_obs_num = 14
        self.action_space = spaces.Box(low=-1., high=1., shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        low_level_env = LowLevelEnv()
        self.low_level_model = SAC.load("models/model_env2.pkl", env=low_level_env)
        self.low_level_obs = low_level_env.reset()

        self.car_dev_before = np.array([0, 0])
        self.traj_before = np.array([[3 * i, -8.0525] for i in range(5)])

        pygame.init()
        self.screen = pygame.display.set_mode((self.road.road_length * 10, - self.road.road_width * 10))
        pygame.display.set_caption("B level Environment")

        self.traj_data = np.array([3, -8.0525])

    def _initial_state(self):
        self.time = 0
        self.car.reset_car()
        return np.zeros(self.observation_space.shape)

    def reset(self):
        self.reset_num += 1
        self.render()
        return self._initial_state()

    def close(self):
        pygame.quit()

    def step(self, action):
        # 기존 action : steering vel -> scalar
        # Policy b action : traj points -> array(list)
        # low_level_obs에 cary, carv 들어가고, car_dev랑 lookahead는 action(신규)에서 가져온 trj 정보로
        done = False
        self.test_num += 1

        traj_lowlevel_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, self.traj_before).flatten()
        self.low_level_obs = np.concatenate((np.array([self.car.carv]), traj_lowlevel_rel))
        steering_changes = self.low_level_model.predict(self.low_level_obs)

        self.car.move_car(steering_changes[0])

        traj_abs = self.make_trajectory(action)
        self.traj_data = np.vstack((self.traj_data, traj_abs))
        traj_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, traj_abs).flatten()
        car_dev = self.calculate_dev()
        cone_state = self.road.cones_arr[self.road.cones_arr[:, 0] > self.car.carx][:2].flatten()
        state = np.concatenate((traj_rel, cone_state)) # <- Policy B의 state

        if self.road.is_car_in_road(self.car) == 1:
            done = True
        reward = self.getReward(car_dev)
        info = {"carx": self.car.carx, "cary": self.car.cary, "caryaw": self.car.caryaw}

        if self.test_num % 100 == 0:
#            print(f"[Time: {self.time}] [reward: {round(reward, 2)}] [Car pos : {round(self.car.carx, 2), round(self.car.cary, 2)}]")
            print(f"[Time: {self.time}] [reward: {round(reward, 2)}] [Car dev : {round(car_dev[0], 2), round(car_dev[1], 2)}]")

        self.time += 0.01
        self.car_dev_before = car_dev
        self.traj_before = traj_abs

        self.render()

        return state, reward, done, info

    def make_trajectory(self, action):
        def rotate(x1, y1, angle):
            x1 -= self.car.carx
            y1 -= self.car.cary
            x_new = x1 * math.cos(angle) - y1 * math.sin(angle)
            y_new = x1 * math.sin(angle) + y1 * math.cos(angle)
            x_new += self.car.carx
            y_new += self.car.cary
            return (x_new, y_new)
        traj_arr = []
        for idx, angle in enumerate(action):
            traj = np.array([self.car.carx + idx * 3, self.car.cary])
            traj_arr.append(rotate(traj[0], traj[1], angle))

        traj_interpolated1 = np.array([(traj_arr[0][0] + traj_arr[1][0]) / 2, (traj_arr[0][1] + traj_arr[1][1]) / 2])
        traj_interpolated2 = np.array([((traj_arr[1][0] + traj_arr[2][0]) / 2, (traj_arr[1][1] + traj_arr[2][1]) / 2)])
        traj_arr = np.insert(traj_arr, 1, traj_interpolated1, axis=0)
        traj_arr = np.insert(traj_arr, 3, traj_interpolated2, axis=0)
        return traj_arr

    def calculate_dev(self):
        f = interp1d(self.traj_data[:, 0], self.traj_data[:, 1])
        xnew = np.arange(self.traj_data[0][0], self.traj_data[-1][0], 0.01)
        ynew = f(xnew)
        arr = np.array(list(zip(xnew, ynew)))
        distances = np.sqrt(np.sum((arr - [self.car.carx, self.car.cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]
        if arr[dist_index][0] - arr[dist_index - 1][0] == 0:
            devAng2 = np.arctan(np.inf)
        else:
            devAng2 = np.arctan((arr[dist_index][1] - arr[dist_index - 1][1]) / (arr[dist_index][0] - arr[dist_index - 1][0]))
        devAng = - devAng2 - self.car.caryaw
        return devDist, devAng

    def getReward(self, state):
        forbidden_reward, collision_reward, dist_reward, ang_reward = 0, 0, 0, 0
        if self.road.is_car_in_forbidden_area(self.car):
            forbidden_reward = -10000
        if self.road.is_car_colliding_with_cones(self.car):
            collision_reward = -1000
        dist_reward = - abs(state[0]) * 1000
        ang_reward = - abs(state[1]) * 5000

        e = forbidden_reward + collision_reward + dist_reward + ang_reward
        return e

    def to_relative_coordinates(self, carx, cary, caryaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - carx
            dy = point[1] - cary

            rotated_x = dx * np.cos(-caryaw) - dy * np.sin(-caryaw)
            rotated_y = dx * np.sin(-caryaw) + dy * np.cos(-caryaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((128, 128, 128))

        for cone in self.road.cones_shape:
            x, y = cone.centroid.coords[0]
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * 10), int(-y * 10)), 5)

        for trajx, trajy in self.traj_before:
            pygame.draw.circle(self.screen, (0, 128, 0), (trajx * 10, - trajy * 10), 5)

        # Create a Surface for the car.
        car_color = (255, 0, 0)

        half_length = self.car.length * 10 / 2.0
        half_width = self.car.width * 10 / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        # Rotate the car corners
        rotated_corners = []
        for x, y in corners:
            x_rot = x * np.cos(-self.car.caryaw) - y * np.sin(-self.car.caryaw) + self.car.carx * 10
            y_rot = x * np.sin(-self.car.caryaw) + y * np.cos(-self.car.caryaw) - self.car.cary * 10
            rotated_corners.append((x_rot, y_rot))

        # Draw the car on the main screen using the rotated corners
        pygame.draw.polygon(self.screen, car_color, rotated_corners)

        pygame.display.flip()


if __name__ == "__main__":
    env = CarEnv()
    """
    env.car.carx, env.car.cary, env.car.caryaw = 68, -5.0525, 0
    traj_abs = env.find_lookahead_traj(env.car.carx, env.car.cary, [3 * i for i in range(5)])
    traj_rel = env.to_relative_coordinates(env.car.carx, env.car.cary, env.car.caryaw, traj_abs)
    devdist, devAng = env.calculate_dev()
    print(traj_rel)
    print(f"dst: {devdist}, ang: {devAng}")
    """
    model = SAC("MlpPolicy", env, verbose=1)
    try:
        model.learn(total_timesteps=10000 * 300)
    except KeyboardInterrupt:
        print("Learning interrupted. Will save the model now.")
    finally:
        print("Env test Finished.")
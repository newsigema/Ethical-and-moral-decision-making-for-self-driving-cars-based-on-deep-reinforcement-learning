import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# =======================================================================================================================

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "modeltrainingv2"

MEMORY_FRACTION = 0.4
MIN_REWARD = -0.65

EPISODES = 100000

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.99975  #
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


# =======================================================================================================================

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# =======================================================================================================================
#  Tensorboard
class ModifiedTensorBoard(TensorBoard):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)


    def set_model(self, model):
        pass


    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)


    def on_batch_end(self, batch, logs=None):
        pass


    def on_train_end(self, _):
        pass


    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# =======================================================================================================================

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    else_actor_type = 'road'

    def __init__(self):
        self.client = carla.Client(host='localhost', port=2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world('Town02_Opt',
                                            carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        # self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.the_people = random.choice(self.blueprint_library.filter('walker.pedestrian.0001'))
        self.another_vehicle_bp = random.choice(self.blueprint_library.filter("vehicle"))

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.hit_the_vehicle = False
        self.hit_the_person = False

        self.transform = carla.Transform(carla.Location(x=-7.530000, y=150.729980, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # =======================================================



        another_transform_1 = carla.Transform(carla.Location(x=-7.530000, y=180.729980, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.another_vehicle_1 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_1)
        self.actor_list.append(self.another_vehicle_1)

        another_transform_2 = carla.Transform(carla.Location(x=193, y=156.729980, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        self.another_vehicle_2 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_2)
        self.actor_list.append(self.another_vehicle_2)

        another_transform_3 = carla.Transform(carla.Location(x=176.9, y=109.5, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        self.another_vehicle_3 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_3)
        self.actor_list.append(self.another_vehicle_3)

        another_transform_4 = carla.Transform(carla.Location(x=32, y=105, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.another_vehicle_4 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_4)
        self.actor_list.append(self.another_vehicle_4)

        another_transform_5 = carla.Transform(carla.Location(x=114, y=302.7, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.another_vehicle_5 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_5)
        self.actor_list.append(self.another_vehicle_5)

        another_transform_6 = carla.Transform(carla.Location(x=17.7, y=306.2, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.another_vehicle_6 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_6)
        self.actor_list.append(self.another_vehicle_6)

        another_transform_8 = carla.Transform(carla.Location(x=190, y=260.3, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        self.another_vehicle_8 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_8)
        self.actor_list.append(self.another_vehicle_8)

        another_transform_9 = carla.Transform(carla.Location(x=158, y=237.3, z=0.500000),
                                              carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.another_vehicle_9 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_9)
        self.actor_list.append(self.another_vehicle_9)

        another_transform_10 = carla.Transform(carla.Location(x=193, y=237.3, z=0.500000),
                                               carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        self.another_vehicle_10 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_10)
        self.actor_list.append(self.another_vehicle_10)

        another_transform_11 = carla.Transform(carla.Location(x=42.1, y=281.3, z=0.500000),
                                               carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        self.another_vehicle_11 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_11)
        self.actor_list.append(self.another_vehicle_11)

        another_transform_12 = carla.Transform(carla.Location(x=179.6, y=307.1, z=0.500000),
                                               carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
        self.another_vehicle_12 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_12)
        self.actor_list.append(self.another_vehicle_12)

        another_transform_13 = carla.Transform(carla.Location(x=41.4, y=213.0, z=0.500000),
                                               carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        self.another_vehicle_13 = self.world.spawn_actor(self.another_vehicle_bp, another_transform_13)
        self.actor_list.append(self.another_vehicle_13)
        # =======================================================


        the_person_transform_2 = carla.Transform(carla.Location(x=-9.8, y=310.2, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_2 = self.world.spawn_actor(self.the_people, the_person_transform_2)
        self.actor_list.append(self.the_person_2)

        the_person_transform_3 = carla.Transform(carla.Location(x=11, y=314.4, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_3 = self.world.spawn_actor(self.the_people, the_person_transform_3)
        self.actor_list.append(self.the_person_3)

        the_person_transform_4 = carla.Transform(carla.Location(x=55.2, y=310.3, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_4 = self.world.spawn_actor(self.the_people, the_person_transform_4)
        self.actor_list.append(self.the_person_4)

        the_person_transform_5 = carla.Transform(carla.Location(x=105.1, y=311.4, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_5 = self.world.spawn_actor(self.the_people, the_person_transform_5)
        self.actor_list.append(self.the_person_5)

        the_person_transform_6 = carla.Transform(carla.Location(x=34.5, y=182.3, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_6 = self.world.spawn_actor(self.the_people, the_person_transform_6)
        self.actor_list.append(self.the_person_6)

        the_person_transform_7 = carla.Transform(carla.Location(x=10.7, y=180.4, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_7 = self.world.spawn_actor(self.the_people, the_person_transform_7)
        self.actor_list.append(self.the_person_7)

        the_person_transform_8 = carla.Transform(carla.Location(x=-0.1, y=215.6, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_8 = self.world.spawn_actor(self.the_people, the_person_transform_8)
        self.actor_list.append(self.the_person_8)

        the_person_transform_9 = carla.Transform(carla.Location(x=-13, y=244.3, z=0.500000),
                                                 carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_9 = self.world.spawn_actor(self.the_people, the_person_transform_9)
        self.actor_list.append(self.the_person_9)

        the_person_transform_10 = carla.Transform(carla.Location(x=49.8, y=204.5, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_10 = self.world.spawn_actor(self.the_people, the_person_transform_10)
        self.actor_list.append(self.the_person_10)

        the_person_transform_11 = carla.Transform(carla.Location(x=80.8, y=195, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_11 = self.world.spawn_actor(self.the_people, the_person_transform_11)
        self.actor_list.append(self.the_person_11)

        the_person_transform_12 = carla.Transform(carla.Location(x=165, y=196, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_12 = self.world.spawn_actor(self.the_people, the_person_transform_12)
        self.actor_list.append(self.the_person_12)

        the_person_transform_13 = carla.Transform(carla.Location(x=185.4, y=161.6, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_13 = self.world.spawn_actor(self.the_people, the_person_transform_13)
        self.actor_list.append(self.the_person_13)

        the_person_transform_14 = carla.Transform(carla.Location(x=150, y=101.2, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_14 = self.world.spawn_actor(self.the_people, the_person_transform_14)
        self.actor_list.append(self.the_person_14)

        the_person_transform_15 = carla.Transform(carla.Location(x=5.9, y=113.9, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_15 = self.world.spawn_actor(self.the_people, the_person_transform_15)
        self.actor_list.append(self.the_person_15)

        the_person_transform_16 = carla.Transform(carla.Location(x=139.3, y=210.9, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_16 = self.world.spawn_actor(self.the_people, the_person_transform_16)
        self.actor_list.append(self.the_person_16)


        the_person_transform_18 = carla.Transform(carla.Location(x=-3, y=181, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_18 = self.world.spawn_actor(self.the_people, the_person_transform_18)
        self.actor_list.append(self.the_person_18)

        the_person_transform_19 = carla.Transform(carla.Location(x=-12.4, y=138, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_19 = self.world.spawn_actor(self.the_people, the_person_transform_19)
        self.actor_list.append(self.the_person_19)

        the_person_transform_20 = carla.Transform(carla.Location(x=152, y=80.3, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_20 = self.world.spawn_actor(self.the_people, the_person_transform_20)
        self.actor_list.append(self.the_person_20)


        the_person_transform_22 = carla.Transform(carla.Location(x=169, y=89, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_22 = self.world.spawn_actor(self.the_people, the_person_transform_22)
        self.actor_list.append(self.the_person_22)

        the_person_transform_23 = carla.Transform(carla.Location(x=212, y=93.7, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_23 = self.world.spawn_actor(self.the_people, the_person_transform_23)
        self.actor_list.append(self.the_person_23)

        the_person_transform_24 = carla.Transform(carla.Location(x=202, y=123, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_24 = self.world.spawn_actor(self.the_people, the_person_transform_24)
        self.actor_list.append(self.the_person_24)

        the_person_transform_25 = carla.Transform(carla.Location(x=198, y=199, z=0.500000),
                                                  carla.Rotation(pitch=0.000000, yaw=89.999954, roll=0.000000))
        self.the_person_25 = self.world.spawn_actor(self.the_people, the_person_transform_25)
        self.actor_list.append(self.the_person_25)

        # =======================================================

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(carla.Location(x=-5, z=3))  # 已经被调整为较合理的摄像机位
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)


        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)
        self.hit_the_vehicle = isinstance(event.other_actor, carla.Vehicle)
        self.hit_the_person = isinstance(event.other_actor, carla.Walker)
        self.else_actor_type = get_actor_display_name(event.other_actor)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))


        if self.hit_the_person == True and len(self.collision_hist) != 0 and kmh >50:
            done = True
            reward = -1.1
        elif self.hit_the_person == True and len(self.collision_hist) != 0 and kmh <=50 and kmh>20:
            done = True
            reward = -1.0
        elif self.hit_the_person == True and len(self.collision_hist) != 0 and  kmh <= 20:
            done = True
            reward = -1.05
        elif self.hit_the_vehicle == True and len(self.collision_hist) != 0 and kmh >50:
            done = True
            reward = -0.3
        elif self.hit_the_vehicle == True and len(self.collision_hist) != 0 and kmh <=50 and kmh>20:
            done = True
            reward = -0.2
        elif self.hit_the_vehicle == True and len(self.collision_hist) != 0 and  kmh <= 20:
            done = True
            reward = -0.25
        elif len(self.collision_hist) != 0 and kmh >50:
            done = True
            reward = -0.6
        elif len(self.collision_hist) != 0 and kmh <=50 and kmh>20:
            done = True
            reward = -0.5
        elif len(self.collision_hist) != 0 and kmh <= 20:
            done = True
            reward = -0.55
        elif kmh >= 50:
            done = False
            reward = -0.1
        elif kmh < 20:
            done = False
            reward = -0.05
        else:
            done = False
            reward = 0

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


# =======================================================================================================================

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):

        self.replay_memory.append(transition)
        self.replay_memory=sorted(self.replay_memory,key=lambda x:x[2],reverse=False)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = np.random.choice(a=self.replay_memory, size=MINIBATCH_SIZE,p = lambda x:np.abs(x[2])/100)


        current_states = np.array([transition[0] for transition in minibatch]) / 255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                           callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


# =======================================================================================================================

if __name__ == '__main__':
    FPS = 60

    ep_rewards = [-200]


    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


    if not os.path.isdir('models'):
        os.makedirs('models')


    agent = DQNAgent()
    env = CarEnv()


    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)


    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))


    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):


        env.collision_hist = []

        agent.tensorboard.step = episode

        episode_reward = 0
        step = 1

        current_state = env.reset()

        done = False
        episode_start = time.time()

        while True:


            if np.random.random() > epsilon:

                action = np.argmax(agent.get_qs(current_state))
            else:

                action = np.random.randint(0, 3)

                time.sleep(1 / FPS)

            new_state, reward, done, _ = env.step(action)


            episode_reward += reward


            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break


        for actor in env.actor_list:
            actor.destroy()


        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)


            if min_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


    agent.terminate = True
    trainer_thread.join()
    agent.model.save(
        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
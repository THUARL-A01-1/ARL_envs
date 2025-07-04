import cv2
import gc
import gymnasium as gym
from io import BytesIO
import mujoco
import matplotlib.pyplot as plt
import numpy as np
import os
import time

class AllegroEnv(gym.Env):
    def __init__(self):
        """
        AllegroEnv is an implementation of the Allegro + Tac3D engineed by Mujoco, with API formulated based on Gym.
        AllegroEnv supports the following important methods:
        - step(): Take an action coltrolled by velocity in position loop.
        - reset(): Reset the environment and return the initial observation.
        - render(): Render the current snapshot or the whole episode.
        - close(): Close the environment and release resources.
        """
        self.model_path = os.path.join('allegro', 'scene.xml')
        with open(self.model_path,"r") as f:
            self.xml_content = f.read()
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml_content)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_renderer = mujoco.Renderer(self.mj_model, 480, 640)
        self.joint_dict = {self.mj_model.joint(i).name: i for i in range(self.mj_model.njnt)}
        self.actuator_dict = {self.mj_model.actuator(i).name: i for i in range(self.mj_model.actuator_actnum.shape[0])}
        
        self.max_iter, self.pos_tolerane, self.velocity_tolerance, self.force_tolerance = 1000, 0.001, 0.001, 1  # Maximum iteration and error tolerance of position error
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.mj_model.actuator_actnum.shape[0],), dtype=np.float32)  # Action space is a 5D vector
        self.observation_space = gym.spaces.Dict({
            "visual": gym.spaces.Box(low=0, high=255, shape=(3, 640, 480), dtype=np.uint8),
            "tactile_th": gym.spaces.Box(low=-1, high=1, shape=(3, 20, 20), dtype=np.float32),
            "tactile_if": gym.spaces.Box(low=-1, high=1, shape=(3, 20, 20), dtype=np.float32),
            "tactile_mf": gym.spaces.Box(low=-1, high=1, shape=(3, 20, 20), dtype=np.float32),
            "tactile_pf": gym.spaces.Box(low=-1, high=1, shape=(3, 20, 20), dtype=np.float32),
            "joint": gym.spaces.Box(low=-1, high=1, shape=self.mj_data.qpos.shape, dtype=np.float32)}  # joint: 29D = hand translation (3D) + hand rotation (3D) + th finger (4D) + if finger (4D) + mf finger (4D) + pf finger (4D) + object free joint (3D translation + 4D quaternion rotation)
        )  # Observation space
        
        self.episode_buffer = {"visual": [], "tactile_th": [], "tactile_if": [], "tactile_mf": [], "tactile_pf": [], "joint": []}  # Episode buffer for replay
        self.episode_mode = "keyframe"  # Full mode for enhancing the display, keyframe mode for training
        self.render_mode = "episode"  # snapshot mode for rendering the current frame, episode mode for rendering the whole episode
        # self.reset()

    def reset(self):
        del self.mj_model
        del self.mj_data
        gc.collect()
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml_content)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.episode_buffer = {"visual": [], "tactile_th": [], "tactile_if": [], "tactile_mf": [], "tactile_pf": [], "joint": []}
        return self.mj_renderer.render()

    def add_frame(self):
        self.mj_renderer.update_scene(self.mj_data, camera="main")
        th_tactile = self.mj_data.sensordata[:1200].copy().reshape(3, 20, 20)
        if_tactile = self.mj_data.sensordata[1200:2400].copy().reshape(3, 20, 20)
        mf_tactile = self.mj_data.sensordata[2400:3600].copy().reshape(3, 20, 20)
        pf_tactile = self.mj_data.sensordata[3600:4800].copy().reshape(3, 20, 20)
        self.episode_buffer["visual"].append(self.mj_renderer.render())
        self.episode_buffer["tactile_th"].append(th_tactile)
        self.episode_buffer["tactile_if"].append(if_tactile)
        self.episode_buffer["tactile_mf"].append(mf_tactile)
        self.episode_buffer["tactile_pf"].append(pf_tactile)
        self.episode_buffer["joint"].append(self.mj_data.qpos.copy())

    def step(self, action, sleep=False):
        """
        Take an action by position control in velocity loop, with a PD controller.
        :param 
            action: 7D vector representing the relative position change or target force.
            sleep: if True, then the iteration number is fixed to max_iter.
        :return: observation, reward, done, info
        """
        target_pos, target_force = self.mj_data.qpos[0:6].copy() + action[0:6], action[6]
        for iter in range(self.max_iter):
            error_pos = target_pos - self.mj_data.qpos[0:6].copy()
            velocity = self.mj_data.qvel[self.mj_model.jnt_qposadr[0:6]]
            self.mj_data.ctrl[0:6] = np.clip(20 * error_pos, -10, 10)  # PD controller
            # if error_pos[2] > 1e-3:  # If the hand is lifted, add a feedforward item
            #     self.mj_data.ctrl[2] += 0.019 * np.sign(error_pos[2])
            
            current_force = np.array([self.mj_data.sensordata[0:1200:3].sum(), 
                             self.mj_data.sensordata[1200:2400:3].sum(), 
                             self.mj_data.sensordata[2400:3600:3].sum(),
                             self.mj_data.sensordata[3600:4800:3].sum()])
            error_force = np.array([target_force, target_force / 3, target_force / 3, target_force / 3]) - current_force
            self.mj_data.ctrl[6:10] = current_force + max(0.2, pow(iter/self.max_iter, 0.5)) * error_force
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            if self.episode_mode == "full" and iter % 50 == 0:  # Add frames every 50 iterations in full mode
                print(f"Iteration {iter}, error_pos: {np.round(error_pos, 2)}, error_force: {np.round(error_force, 2)}")
                self.add_frame()
            if (not sleep) and np.linalg.norm(error_pos) < self.pos_tolerane and np.linalg.norm(velocity) < self.velocity_tolerance and np.linalg.norm(error_force) < self.force_tolerance:  # Break if the error is smaller than the tolerance
                break
        
        if self.episode_mode == "keyframe":  # Only keep the last frame in keyframe mode
            print(f"Iteration {iter}, error_pos: {np.round(error_pos, 2)}, error_force: {np.round(error_force, 2)}")
            self.add_frame()
        
        return self.get_observation(), self.get_reward(), self.get_done(), {}
    
    def get_observation(self):
        return {
            "visual": self.episode_buffer["visual"][-1],
            "tactile_th": self.episode_buffer["tactile_th"][-1],
            "tactile_if": self.episode_buffer["tactile_if"][-1],
            "tactile_mf": self.episode_buffer["tactile_mf"][-1],
            "tactile_pf": self.episode_buffer["tactile_pf"][-1],
            "joint": self.episode_buffer["joint"][-1]}  # Return the last frame in the episode buffer
    
    def get_reward(self):
        return 0
    
    def get_done(self):
        return False
    
    def draw_tactile(self, tactile):
        """
        Draw tactile sensor data (20*20*3), with the color representing the z-force.
        Note: The x-axis in the finger coordinate system is the gravity direction in the world.
        """
        X, Y = np.meshgrid(np.arange(20), np.arange(20))
        Fx, Fy, Fz = tactile[1, ...], tactile[2, ...], tactile[0, ...]
        quiver = plt.quiver(Y, X, Fy, Fx, Fz, cmap='coolwarm', pivot='tail',
                            scale=1, width=0.005, headwidth=4, headlength=6, headaxislength=4)
        plt.axis('off')
        plt.axis('equal')
        # plt.colorbar(quiver, label='fn/N')
        
        # Save the plot to a buffer and convert it to a cv2 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buffer.seek(0)
        image = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        buffer.close()

        return image[:, 68:68+360, :]  # Crop the image to remove the white margin

    def draw_frame(self, frame_id):
        """
        Draw a specific frame in the episode buffer.
        :param frame_id: the index of the frame in the episode buffer.
        """
        visual_frame = self.episode_buffer["visual"][frame_id]
        visual_frame = cv2.resize(visual_frame, (720, 540), interpolation=cv2.INTER_LINEAR)
        tactile_left_frame = self.draw_tactile(self.episode_buffer["tactile_th"][frame_id])
        tactile_right_frame = self.draw_tactile(self.episode_buffer["tactile_if"][frame_id])
        tactile_combined = np.hstack((tactile_left_frame, tactile_right_frame))
        combined_frame = np.vstack((visual_frame, tactile_combined))
        
        return combined_frame

    def render(self):
        """
        Render the current snapshot or the whole episode.
        - snapshot mode: render the current snapshot.
        - episode mode: render the whole episode.     
        """
        if self.render_mode == "snapshot":  # Render the current snapshot
            frame = self.draw_frame(-1)
            cv2.imshow('simulation img', frame)
            cv2.waitKey(0)
        elif self.render_mode == "episode":
            video_frames = []
            for frame_id in range(len(self.episode_buffer["visual"])):
                frame = self.draw_frame(frame_id)
                video_frames.append(frame)
            while True:
                for frame in video_frames:
                    cv2.imshow('Episode Playback', frame)
                    if cv2.waitKey(500) & 0xFF == ord('q'):  # 按 'q' 键退出
                        cv2.destroyAllWindows()
                        return
                time.sleep(0.5)


        
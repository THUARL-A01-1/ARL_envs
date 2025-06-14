from allegro.allegro import AllegroEnv
import mujoco
import mujoco.viewer as viewer
import numpy as np
import os


def test_in_GUI():
    model_path = os.path.join('allegro', 'scene.xml')
    with open(model_path,"r",encoding='utf-8') as f:
        xml_content = f.read()
    model = mujoco.MjModel.from_xml_string(xml_content)
    model.body_gravcomp[:] = 1.0
    mj_data = mujoco.MjData(model)
    viewer.launch(model, mj_data)


def test_env():
    env = AllegroEnv()
    _ = env.reset()
    env.step(np.array([0, 0, 0, 0, 0, 0, 0]))
    env.step(np.array([0, 0, 0, 0, 0, 0, 3]))
    env.step(np.array([0, 0, 0, 0, 0, 0, 3]))
    env.step(np.array([0, 0, 0.05, 0, 0, 0, 3]))
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    env.mj_model.body_gravcomp[body_id] = 0.0
    env.step(np.array([0, 0, 0.05, 0, 0, 0, 3]), sleep=False)
    env.render()






if __name__ == "__main__":
    # test_in_GUI()
    test_env()
import mujoco
import mujoco.viewer as viewer
import os


def test_in_GUI():
    model_path = os.path.join('allegro', 'scene.xml')
    with open(model_path,"r",encoding='utf-8') as f:
        xml_content = f.read()
    model = mujoco.MjModel.from_xml_string(xml_content)
    model.body_gravcomp[:] = 1.0
    mj_data = mujoco.MjData(model)
    viewer.launch(model, mj_data)









if __name__ == "__main__":
    test_in_GUI()
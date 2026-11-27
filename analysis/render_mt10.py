# Render one frame per MT10 task (run where metaworld + MuJoCo EGL are available); output used by static_figs.py.
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, metaworld
from PIL import Image
mt10 = metaworld.MT10(seed=17719)
for name, cls in mt10.train_classes.items():
    task = [t for t in mt10.train_tasks if t.env_name == name][0]
    try:
        env = cls(render_mode="rgb_array", camera_name="corner", width=480, height=480)
    except TypeError:
        env = cls(render_mode="rgb_array", camera_name="corner")
    env.set_task(task); env.reset()
    for _ in range(3): env.step(np.zeros(env.action_space.shape))
    img = env.render()
    Image.fromarray(np.flipud(np.asarray(img))).save(f"{name}.png")
    print(name, np.asarray(img).shape)
    env.close()

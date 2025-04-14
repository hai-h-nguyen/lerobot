
import numpy as np
import wandb

run = wandb.init()
# axes are (time, channel, height, width)
# frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
video_path = "output/peg-insertion-triangle-front-view-no-ft/eval/videos_step_020000/eval_episode_3.mp4"
wandb_video = wandb.Video(video_path, fps=10, format="mp4")
run.log({"video": wandb.Video(video_path)})
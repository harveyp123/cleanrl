# import json

# run_id = "run-20231019_143917-ftgx4zi6"
# wandb_file = f"wandb/{run_id}/run-ftgx4zi6.wandb"
# with open(wandb_file, 'r', errors='ignore') as f:
#     wandb_data = json.load(f)
# pass


import wandb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

api_key = "ca2f2a2ae6e84e31bbc09a8f35f9b9a534dfbe9b"
wandb.login(key=api_key)
username = "jincan333"
project = "ensemble_distill_unequal_restart_atari"
alpha5_run_id_list = ["2bfym30s", "2xzz0fex", "9je4gn16", "cy0mg605"]
alpha0_run_id_list = ["ftgx4zi6", "ixu2mtc0", "sb1dpfbd"]
x_list = []
y_list = []

def extract_and_interpolate(run_id_list):
    step_list=[]
    score_list=[]
    smooth_step_list = []
    smooth_score_list = []
    for i, run_id in enumerate(run_id_list):
        run = wandb.Api().run(f"{username}/{project}/{run_id}")
        metrics = run.scan_history()
        step_list.append([])
        score_list.append([])
        for metric_data in metrics:
            if metric_data['teacher avg return'] is not None:
                step_list[i].append(metric_data['_step'])
                score_list[i].append(metric_data['teacher avg return'])
        smooth_step = np.arange(0, max(step_list[i])+1)
        smooth_score = np.interp(smooth_step, np.array(step_list[i]), np.array(score_list[i]))
        smooth_step_list.append(smooth_step)
        smooth_score_list.append(smooth_score)
        data = pd.DataFrame({'Step': smooth_step, 'Score': smooth_score})
        data.to_csv(f'plot/{run_id}.csv', index=None)
    smooth_score_avg = np.mean(smooth_score_list, axis = 0)
    return smooth_step*8, smooth_score_avg

for run_id_list in [alpha5_run_id_list, alpha0_run_id_list]:
    smooth_step, smooth_score_avg = extract_and_interpolate(run_id_list)
    x_list.append(smooth_step)
    y_list.append(smooth_score_avg)

title = 'BreakoutNoFrameskip-v4, Scores'
legends=['alpha = 0.5', 'alpha = 0']
for x, y, legend in zip(x_list, y_list, legends):
    plt.plot(x, y, label=legend)
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.show()
plt.savefig(f'plot/{title}.pdf')

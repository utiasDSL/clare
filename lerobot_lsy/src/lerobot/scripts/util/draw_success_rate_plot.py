# import matplotlib.pyplot as plt

# # Data

# x = [0, 1, 2, 3, 4, 5, 6]
# task5_success_rate = [0, 95, 0, 0, 0, 0, 0]
# task6_success_rate = [0, 0, 47, 0, 0, 0, 0]
# task7_success_rate = [0, 0, 0, 81, 0, 0, 0]
# task8_success_rate = [0, 0, 0, 0, 63, 0, 0]
# task9_success_rate = [0, 0, 0, 0, 0, 88, 88]

# title = 'Adapter Switch'




# plt.figure(figsize=(15, 5))
# l1, = plt.plot(x, task5_success_rate, linestyle='-', label='Task 5')
# plt.fill_between(x, task5_success_rate, alpha=0.2, color=l1.get_color())

# l2, = plt.plot(x, task6_success_rate, linestyle='-', label='Task 6')
# plt.fill_between(x, task6_success_rate, alpha=0.2, color=l2.get_color())

# l3, = plt.plot(x, task7_success_rate, linestyle='-', label='Task 7')
# plt.fill_between(x, task7_success_rate, alpha=0.2, color=l3.get_color())

# l4, = plt.plot(x, task8_success_rate, linestyle='-', label='Task 8')
# plt.fill_between(x, task8_success_rate, alpha=0.2, color=l4.get_color())

# l5, = plt.plot(x, task9_success_rate, linestyle='-', label='Task 9')
# plt.fill_between(x, task9_success_rate, alpha=0.2, color=l5.get_color())

# plt.xlabel('Number of Learned Tasks', fontsize=24, loc="right")
# plt.ylabel('Success Rate [%]', fontsize=24, loc="top")
# # plt.title(title, fontsize=20)
# plt.xticks(x[:-1], fontsize=18)
# plt.yticks(range(0, 101, 20), fontsize=18)
# plt.legend(fontsize=15, loc='upper left')
# plt.grid(axis='y', linestyle='--', alpha=0.7)  # Enable horizontal grid lines


# ax = plt.gca()
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)

# plt.tight_layout()
# plt.show()

import os
import json
import matplotlib.pyplot as plt

# Parameters
base_dir = "./outputs/eval"
job_name = "eval_cl_adapter_switch_ditflow_cond_mlp_libero_10"  # <-- change to your job name
num_tasks = 10  # assuming tasks go from 0 to 9
allow_negative_nbt = False

# manual_input
task_success_rates = {
    0:[58, 61, 59, 71, 60, 63, 59, 64, 66, 65],
    1:[0,  88, 88, 87, 86, 87, 90, 92, 94, 85],
    2:[0,  0,  93, 95, 93, 89, 97, 93, 94, 95],
    3:[0,  0,  0,  87, 94, 93, 92, 93, 89, 87],
    4:[0,  0,  0,  0,  53, 54, 53, 51, 55, 51],
    5:[0,  0,  0,  0,  0,  89, 92, 91, 91, 89],
    6:[0,  0,  0,  0,  0,  0,  69, 53, 60, 58],
    7:[0,  0,  0,  0,  0,  0,  0,  80, 84, 80],
    8:[0,  0,  0,  0,  0,  0,  0,  0,  73, 66],
    9:[0,  0,  0,  0,  0,  0,  0,  0,  0,  80],
}

# Collect success rates for each task across different checkpoints
# task_success_rates = {i: [] for i in range(num_tasks)}

# # Loop over all checkpoints (task_0, task_1, ...)
# for checkpoint in range(num_tasks):
#     eval_dir = os.path.join(base_dir, f"{job_name}_task_{checkpoint}")
#     json_path = os.path.join(eval_dir, "multitask_eval_info.json")

#     if not os.path.exists(json_path):
#         # If file doesn't exist, just append zeros
#         for i in range(num_tasks):
#             task_success_rates[i].append(0)
#         continue

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     # For each task, try to get pc_success
#     for i in range(num_tasks):
#         task_key = f"Libero_10_Task_{i}"
#         if task_key in data.get("per_task", {}):
#             pc_success = data["per_task"][task_key].get("pc_success", 0)
#             task_success_rates[i].append(pc_success)
#         else:
#             task_success_rates[i].append(0)

for i in range(num_tasks):
    task_success_rates[i].append(task_success_rates[i][-1])




# Prepare x axis (number of learned tasks)
x = list(range(num_tasks + 2))  # 0 .. 10

# Plot
plt.figure(figsize=(17, 8))
lines = []
for i in range(num_tasks):
    rates = [0] + task_success_rates[i]  # prepend 0 for alignment
    line, = plt.plot(x, rates, linestyle='-', label=f"Task {i}")
    plt.fill_between(x, rates, alpha=0.2, color=line.get_color())
    lines.append(line)

plt.xlabel('# Learned Tasks', fontsize=24, loc="right")
plt.ylabel('Success Rate [%]', fontsize=24, loc="top")
plt.xticks(x[:-1], fontsize=18)
plt.yticks(range(0, 101, 20), fontsize=18)
plt.legend(fontsize=20, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
# plt.show()

plt.savefig(f"./outputs/figs/{job_name}.png")

import csv

# Sort keys numerically to ensure correct task order
tasks = sorted(task_success_rates.keys(), key=lambda x: int(x))

# Create CSV file
with open(f"./outputs/csv/{job_name}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    
    # Write header
    header = [f"task {t}" for t in tasks]
    writer.writerow(header)
    
    # Write rows (max 10 rows)
    for i in range(10):
        row = [task_success_rates[t][i] if t <= i else "-" for t in tasks]
        writer.writerow(row)

print(f"CSV file '{job_name}.csv' created.")

import numpy as np



success_rate_list = []

for key in task_success_rates:
    success_rate_list.append(task_success_rates[key][:-1])

success_rate = np.array(success_rate_list).astype(int)

# FWT: average of the diagonal elements
fwt = np.mean(np.diag(success_rate))

# NBT: average of the differences between i-th value and the following value in row i
nbt_values_per_task = []
rnbt_values_per_task = []
for i in range(success_rate.shape[0]-1):
    nbt_values = []
    rnbt_values = []
    for j in range(i + 1, success_rate.shape[1]):
        
        if success_rate[i, i] > 0:
            if allow_negative_nbt:
                diff = success_rate[i, i] - success_rate[i, j]
            else:
                if success_rate[i, i] >= success_rate[i, j]:
                    diff = success_rate[i, i] - success_rate[i, j]
                else:
                    diff = np.int32(0)

            nbt_values.append(diff)
            rnbt_values.append(diff / success_rate[i, i])
        
    
    if len(rnbt_values) > 0:
        nbt_values_per_task.append(np.mean(nbt_values))
        rnbt_values_per_task.append(np.mean(rnbt_values))


nbt = np.sum(nbt_values_per_task) / (len(nbt_values_per_task))
rnbt = np.sum(rnbt_values_per_task) / (len(rnbt_values_per_task)) * 100

# AUC: average of the mean of each row from index i to the end
auc_values = []
for i in range(success_rate.shape[0]):
    row = success_rate[i, i:]  # from i-th to last element in row i
    auc_values.append(np.mean(row))
auc = np.mean(auc_values)

print(f"FWT: {fwt:.2f}")
print(f"NBT: {nbt:.2f}")
print(f"RNBT: {rnbt:.2f}")
print(f"AUC: {auc:.2f}")


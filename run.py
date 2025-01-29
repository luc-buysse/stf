#!/bin/python3

import yaml
import os
import sys
import random

# Set the working directory to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if len(sys.argv) != 2:
    print('Error: this command expects exactly one argument: the name of a training configuration stored in stf/configs')

config_name = sys.argv[1]
config_full = f"configs/{config_name}.yml"

if not os.path.exists(config_full):
    print(f"Error: the specified configuration \"{config_name}\" wasn't found")

with open(config_full, 'r') as f:
    slurm_config = yaml.safe_load(f).get("slurm", dict())
    ngpus = slurm_config.get('ngpus', 2)
    batch_size = slurm_config.get('batch_size', 8)

master_port = random.randint(12345, 12400) 

script_path = f"train_scripts/train_{config_name}.sh"
with open(script_path, 'w') as f:
    f.write(f"""#!/bin/bash
#SBATCH --job-name={config_name}
#SBATCH --output=logs/{config_name}/output.log
#SBATCH --error=logs/{config_name}/error.log
#SBATCH --time=96:00:00
#SBATCH --gpus={ngpus}         # GPUs per task
#SBATCH --gpus-per-task=1
#SBATCH --ntasks={ngpus}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate alice

export WORLD_SIZE={ngpus}

export CUDA_VISIBLE_DEVICES=0,1,2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT={master_port}

# Printing environment variables
echo "MASTER_ADDR="$MASTER_ADDR
echo "NODELIST="$SLURM_NODELIST
echo "WORLD_SIZE="$WORLD_SIZE

# Run the training script
srun --export=ALL python3 train.py -c configs/{config_name}.yml --batch-size {batch_size}
""")

os.system(f'sbatch {script_path}')


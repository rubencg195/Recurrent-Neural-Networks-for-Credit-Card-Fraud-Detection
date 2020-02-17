#!/bin/bash
## SBATCH --gres=gpu:4        # request GPU "generic resource"
## SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=1-00:00      # time (DD-HH:MM)
#SBATCH --output=GRU-0-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --job-name=RNN

#SBATCH --mail-user=rubencg@mun.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module load python/3.6
module load scipy-stack
module load bedtools
module load java

source tensorflow/bin/activate
ipython LOCAL-RNN-Fraud-Detection.py
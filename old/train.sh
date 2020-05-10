#!/bin/bash
## SBATCH --gres=gpu:4        # request GPU "generic resource"
## SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-06:00      # time (DD-HH:MM)
#SBATCH --output=RNN-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --job-name=RNN

#SBATCH --mail-user=rubencg@mun.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source tensorflow/bin/activate    && \
module load python/3.6    && \
module load scipy-stack    && \
module load java    && \
ipython3 LOCAL-RNN-Fraud-Detection.py


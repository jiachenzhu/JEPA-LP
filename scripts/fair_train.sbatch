#!/bin/bash
##SBATCH --partition=scavenge
#SBATCH --partition=learnlab
#SBATCH --job-name=JEPA?
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=448G
#SBATCH --gres=gpu:8
#SBATCH --constraint=volta32gb
#SBATCH --time=10:00:00
#SBATCH --array=1
#SBATCH --signal=B:USR1@60
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --output=logs/train_%a_%j.out
#SBATCH --error=error_logs/train_%a_%j.err

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}

# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

export SLURM_JOB_NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)

source /private/home/jiachenzhu/.miniconda3/etc/profile.d/conda.sh
export PATH=/private/home/jiachenzhu/.miniconda3/bin:$PATH

conda activate
conda activate dev
srun --label python train.py configs/fair_base.yml configs/train_10.yml configs/experiment_$SLURM_ARRAY_TASK_ID.yml
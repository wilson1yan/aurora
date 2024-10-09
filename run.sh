#! /bin/bash

#SBATCH -A CSC529
#SBATCH -t 12:00:00
#SBATCH -p extended
#SBATCH -N 16
#SBATCH -C nvme

unset SLURM_EXPORT_ENV

echo $OUTPUT_FOLDER

module load PrgEnv-gnu/8.5.0
module load rocm/6.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
source deactivate
#source activate /lustre/orion/scratch/wilson/csc529/envs/aurora

#### START OF SBCAST AND CONDA-UNPACK #####

envdir=/lustre/orion/scratch/wilson/csc529/envs
envname=aurora

if [ -d "/mnt/bb/${USER}/${envname}" ]; then
  source activate /mnt/bb/${USER}/${envname}
  echo "/mnt/bb/${USER}/${envname} already exists. skipping sbcast"
else
  echo "copying ${envname} to each node in the job"
  sbcast -pf ${envdir}/${envname}.tar.gz /mnt/bb/${USER}/${envname}.tar.gz
  if [ ! "$?" == "0" ]; then
      # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
      # your application may pick up partially complete shared library files, which would give you confusing errors.
      echo "SBCAST failed!"
      exit 1
  fi

  # Untar the environment file (only need 1 task per node to do this)
  srun -N$SLURM_JOB_NUM_NODES --ntasks-per-node 1 mkdir /mnt/bb/${USER}/${envname}
  echo "untaring torchenv"
  srun -N$SLURM_JOB_NUM_NODES --ntasks-per-node 1 tar -xzf /mnt/bb/${USER}/${envname}.tar.gz -C  /mnt/bb/${USER}/${envname}

  # Unpack the env
  source activate /mnt/bb/${USER}/${envname}
  srun -N$SLURM_JOB_NUM_NODES --ntasks-per-node 1 conda-unpack
fi

#### END OF SBCAST AND CONDA-UNPACK #####
which python

##### START OF SBCAST HF MODEL#####

hfpath=/lustre/orion/world-shared/csc529/wilson/aurora.tar
if [ -d "/mnt/bb/${USER}/huggingface" ]; then
  echo "/mnt/bb/${USER}/hugginface already exists. skipping sbcast"
else
  echo "copying huggingface to each node in the job"
  sbcast -pf ${hfpath} /mnt/bb/${USER}/huggingface.tar
  if [ ! "$?" == "0" ]; then
      # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
      # your application may pick up partially complete shared library files, which would give you confusing errors.
      echo "SBCAST failed!"
      exit 1
  fi

  # Untar the environment file (only need 1 task per node to do this)
  srun -N$SLURM_JOB_NUM_NODES --ntasks-per-node 1 mkdir /mnt/bb/${USER}/huggingface
  echo "untaring hf"
  srun -N$SLURM_JOB_NUM_NODES --ntasks-per-node 1 tar -xf /mnt/bb/${USER}/huggingface.tar -C  /mnt/bb/${USER}
fi

##### END OF SBCAST HF MODEL#####

export MASTER_ADDR=$(hostname -i)
export NCCL_SOCKET_IFNAME=hsn0

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export TORCH_HOME=$WORLDWORK/csc529/$USER
#export HF_HOME=$WORLDWORK/csc529/wilson/huggingface
export HF_HOME=/mnt/bb/${USER}/huggingface
export TRANSFORMERS_OFFLINE=1

export LD_LIBRARY_PATH=/lustre/orion/world-shared/csc529/wilson/code/aws-ofi-rccl2/lib:$LD_LIBRARY_PATH
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN

ranks_per_node=8
gpus_per_rank=$((8/$ranks_per_node))
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))

export PYTHONPATH="$PYTHONPATH:$PWD"

python inference.py \
    --model_path wchai/AuroraCap-7B-VID-xtuner \
    --prompt "Describe the video in detail." \
    --visual_input assets/auroracap/test.mp4 \
    --num_frm 8 \
    --token_kept_ratio 0.8 \
    --temperature 0.0 \
    --top_p 1.0 \
    --num_beams 1 \
    --max_new_tokens 2048

#!/bin/bash
#SBATCH --partition=allgpu,maxgpu
#SBATCH --constraint='A100'|'P100'|'V100'|
#SBATCH --time=72:00:00                           # Maximum time requested
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --chdir=/home/kaechben/slurm_calo_middle        # directory must already exist!
#SBATCH --job-name=hostname
#SBATCH --output=%j.out               # File to which STDOUT will be written
#SBATCH --error=%j.err                # File to which STDERR will be written
#SBATCH --mail-type=NONE                       # Type of email notification- BEGIN,END,FAIL,ALL

unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
module load anaconda3/5.2
. conda-init
conda activate torch_wandb
cd /home/$USER/MF/
python test.py
#python voxel_to_particlecloud.py --files middle_best_w1p_new.hdf5 --in_dir /beegfs/desy/user/kaechben/calochallenge --out_dir /beegfs/desy/user/kaechben/testing --dataset_name MDMA_dataset_2 #example on how to create a particle cloud out of a voxel file
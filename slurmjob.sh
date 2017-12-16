#!/usr/bin/env	bash
#SBATCH	-o slurm3.sh.out
#SBATCH	-p k80
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=04:30:00

python pacman.py -p DeepQAgent -n 50000 -x 50000 -l smallGrid -q

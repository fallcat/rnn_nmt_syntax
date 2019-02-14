#!/bin/bash
#
#SBATCH --job-name=exp01-rnn
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=24
#SBATCH --mem=64GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/wyou/RNN-NMT-Syntax/experiments/exp01/translate_lstm.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wyou@cs.umass.edu

BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/wyou/RNN-NMT-Syntax
EXPERIMENT_PATH=$PROJECT_PATH/experiments/exp01

# Load in python3 and source the venv
module load python3/3.6.6-1810
source $PROJECT_PATH/../py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
PYTHONPATH=$BASE_PATH/wyou/py36/lib/python3.6/site-packages/:$PYTHONPATH

python train.py -l 4

#BASE_PARAMS=( \
#  -d "$BASE_PATH/datasets/wmt/" \
#  -p "$PROJECT_PATH/data/wmt" \
#  --dataset wmt_en_de_parsed \
#  --model parse_transformer \
#  --span 6 \
#  )
#
#env $(cat ~/.comet.ml | xargs) python main.py \
#  "${BASE_PARAMS[@]}" --batch-size 500 \
#  --restore $EXPERIMENT_PATH/checkpoint.pt --average-checkpoints 5 --split test \
#  translate --order-output --output-directory $EXPERIMENT_PATH
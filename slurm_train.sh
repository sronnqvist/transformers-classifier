#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 70:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002026
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "START: $(date)"

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load tensorflow/2.2-hvd
source transformers3.4/bin/activate



export PYTHONPATH=/scratch/project_2002026/samuel/transformer-text-classifier/transformers3.4/lib/python3.7/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MODEL_DIR=/scratch/project_2002026/bert/cased_L-12_H-768_A-12
export MODEL_DIR=/scratch/project_2002026/bert/cased_L-24_H-1024_A-16

export SOURCE_DIR=/scratch/project_2002026/multilabel_bert/neuro_classifier/multilabel

SRC=en
TRG=en
export TRAIN_DIR=data/eacl/$SRC
export DEV_DIR=data/eacl/$TRG
export OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"

LR=3e-5
EPOCHS=6
BS=7

for i in 1 2 3; do
for EPOCHS in 3 4 6; do
for LR in 1e-5 2e-5 8e-5 3e-5 9e-6; do
srun python train.py \
  --model_name jplu/tf-xlm-roberta-large \
  --train $TRAIN_DIR/train.tsv \
  --dev $DEV_DIR/dev.tsv \
  --input_format tsv \
  --lr $LR \
  --seq_len 512 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --output_file $OUTPUT_DIR/model.h5 \
  --log_file logs/train_xlmrL_$SRC-$TRG.tsv
#  --multiclass
#--output_file $OUTPUT_DIR/model.h5 \
#  --load_model $OUTPUT_DIR/model_nblocks3-ep10-2.h5 \
#--load_model $OUTPUT_DIR/model.h5 \
done
done
done

#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"

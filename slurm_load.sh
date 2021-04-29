#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 01:15:00
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

SRC=$1
TRG=$2
export TRAIN_DIR=data/eacl/$SRC
export DEV_DIR=data/eacl/$TRG
export OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"

#MODEL="jplu/tf-camembert-base"
#MODEL="jplu/tf-flaubert-base-cased"
#MODEL="jplu/tf-flaubert-large-cased"
#MODEL="jplu/tf-xlm-roberta-base"
MODEL="jplu/tf-xlm-roberta-large"
#MODEL="bert-base-multilingual-cased"
#MODEL="bert-base-cased"
#MODEL="bert-large-cased"
#MODEL="bert-large-cased-whole-word-masking"
#MODEL="TurkuNLP/bert-base-finnish-cased-v1"
#MODEL="KB/bert-base-swedish-cased"
BS=7
EPOCHS=3
LR=1e-5

#for i in 1 2 3 4 5; do
#for EPOCHS in 3; do
#for LR in 8e-6 7e-6; do
#echo "Settings: src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS"
#echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/experiments.log
echo "Settings: src=$SRC trg=$TRG model_file=$3"

srun python train_load.py \
  --load_model $3 \
  --model_name $MODEL \
  --train $TRAIN_DIR/train.tsv \
  --dev $DEV_DIR/dev.tsv \
  --test $DEV_DIR/test.tsv \
  --input_format tsv \
  --lr $LR \
  --seq_len 512 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --log_file logs/testx_xlmrL_$SRC-$TRG.tsv
#  --output_file $OUTPUT_DIR/model_xlmrL_$SRC-$TRG.h5 \
#  --multiclass
#--output_file $OUTPUT_DIR/model.h5 \
#  --load_model $OUTPUT_DIR/model_nblocks3-ep10-2.h5 \
#--load_model $OUTPUT_DIR/model.h5 \
#done
#done
#done

#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"

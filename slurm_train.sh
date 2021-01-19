#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 23:15:00
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

SRC=sv
TRG=$1
BG="en fi fr"
export TRAIN_DIR=data/eacl/$SRC
export DEV_DIR=data/eacl/$TRG
#export BG_DIR=data/eacl/$BG
export BG_FILES="data/eacl/en/train.tsv data/eacl/fi/train.tsv data/eacl/fr/train.tsv"
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
BGrate=1.0

for i in $4; do
for EPOCHS in $3; do
for LR in $2; do
echo "Settings: src=$SRC trg=$TRG bg=$BG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS"
echo "job=$SLURM_JOBID src=$SRC trg=$TRG bg=$BG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS bg_rate=$BGrate" >> logs/experiments.log
srun python train.py \
  --model_name $MODEL \
  --train $TRAIN_DIR/train.tsv \
  --dev $DEV_DIR/dev.tsv \
  --bg_train "$BG_FILES" \
  --bg_sample_rate $BGrate \
  --input_format tsv \
  --lr $LR \
  --seq_len 512 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --output_file "$OUTPUT_DIR/model_xlmrL_$BG $SRC-$TRG-$i.h5" \
  --log_file "logs/train_xlmrL_$BG $SRC-$TRG.tsv"
#  --test $DEV_DIR/test.tsv \
#  --test_log_file logs/test_xlmrL_$BG++$SRC-$TRG.tsv
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

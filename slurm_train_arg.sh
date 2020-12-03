#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 16:15:00
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

MODEL=$1
MODEL_ALIAS=$2
SRC=$3
TRG=$4
LR_=$5
EPOCHS_=$6
i=$7
BS=7

echo "MODEL:$MODEL"
echo "MODEL_ALIAS:$MODEL_ALIAS"
echo "SRC:$SRC"
echo "TRG:$TRG"
echo "LR:$LR_"
echo "EPOCHS:$EPOCHS_"
echo "i:$i"

export TRAIN_DIR=data/eacl/$SRC
export DEV_DIR=data/eacl/$TRG
export OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"

for EPOCHS in $EPOCHS_; do
for LR in $LR_; do
for j in $i; do
echo "Settings: src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS"
echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/experiments.log
srun python train.py \
  --model_name $MODEL \
  --train $TRAIN_DIR/train.tsv \
  --dev $DEV_DIR/dev.tsv \
  --test $DEV_DIR/test.tsv \
  --input_format tsv \
  --lr $LR \
  --seq_len 512 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --output_file $OUTPUT_DIR/$MODEL_ALIAS-$SRC-$TRG-lr$LR-ep$EPOCHS-$j.h5 \
  --log_file logs/train_$MODEL_ALIAS-$SRC-$TRG.tsv \
  --test_log_file logs/test_$MODEL_ALIAS-$SRC-$TRG.tsv
#  --multiclass
#--output_file $OUTPUT_DIR/model.h5 \
#  --load_model $OUTPUT_DIR/model_nblocks3-ep10-2.h5 \
#--load_model $OUTPUT_DIR/model.h5 \
done
done
done

echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/completed.log


#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"

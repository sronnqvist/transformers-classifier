# transformer-text-classifier

Text classification using Transformers, TensorFlow, and Keras

## Quickstart

Install requirements

    python -m pip install -r requirements.txt

Download sample data

    git clone https://github.com/spyysalo/ylilauta-corpus.git

Train

    python train.py --model_name TurkuNLP/bert-base-finnish-cased-v1 \
        --train ylilauta-corpus/data/ylilauta-train-1000.txt \
        --dev ylilauta-corpus/data/ylilauta-dev.txt

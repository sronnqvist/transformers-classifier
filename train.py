#!/usr/bin/env python3

import sys
import math

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

from transformers import AutoConfig, AutoTokenizer, TFAutoModel
from transformers.optimization_tf import create_optimizer

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning

from readers import READERS, get_reader


# Parameter defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 128
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_PROPORTION = 0.1


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=None,
                    help='pretrained model name')
    ap.add_argument('--train', metavar='FILE', required=True,
                    help='training data')
    ap.add_argument('--dev', metavar='FILE', required=True,
                    help='development data')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=DEFAULT_BATCH_SIZE,
                    help='batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=1,
                    help='number of training epochs')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float, 
                    default=DEFAULT_LR, help='learning rate')
    ap.add_argument('--seq_len', metavar='INT', type=int,
                    default=DEFAULT_SEQ_LEN,
                    help='maximum input sequence length')
    ap.add_argument('--warmup_proportion', metavar='FLOAT', type=float,
                    default=DEFAULT_WARMUP_PROPORTION,
                    help='warmup proportion of training steps')
    ap.add_argument('--input_format', choices=READERS.keys(),
                    default=list(READERS.keys())[0],
                    help='input file format')
    return ap



def load_pretrained(options):
    name = options.model_name
    config = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, config=config)
    model = TFAutoModel.from_pretrained(name, config=config)

    if options.seq_len > config.max_position_embeddings:
        warning(f'--seq_len ({options.seq_len}) > max_position_embeddings '
                f'({config.max_position_embeddings}), using latter')
        options.seq_len = config.max_position_embeddings

    return model, tokenizer, config


def get_optimizer(num_train_examples, options):
    steps_per_epoch = math.ceil(num_train_examples / options.batch_size)
    num_train_steps = steps_per_epoch * options.epochs
    num_warmup_steps = math.floor(num_train_steps * options.warmup_proportion)

    # Mostly defaults from transformers.optimization_tf
    optimizer, lr_scheduler = create_optimizer(
        options.lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_ratio=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay_rate=0.01,
        power=1.0,
    )
    return optimizer


def build_classifier(pretrained_model, num_labels, optimizer, options):
    seq_len = options.seq_len
    input_ids = Input(shape=(seq_len,), dtype='int32')
    token_type_ids = Input(shape=(seq_len,), dtype='int32')
    attention_mask = Input(shape=(seq_len,), dtype='int32')

    pretrained_outputs = pretrained_model([
        input_ids,
        attention_mask,
        token_type_ids
    ])
    pooled_output = pretrained_outputs[1]

    # TODO consider Dropout here
    output_probs = Dense(num_labels)(pooled_output)

    model = Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=[output_probs]
    )

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=[CategoricalAccuracy(name='acc')]
    )

    return model


def load_data(fn, options):
    read = get_reader(options.input_format)
    texts, labels = [], []
    with open(fn) as f:
        for ln, (text, text_labels) in enumerate(read(f, fn), start=1):
            if not text_labels:
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            if len(text_labels) > 1:
                warning(f'multiple labels on line {ln} in {fn}: {l}')
            texts.append(text)
            labels.append(text_labels[0])
    return texts, labels


def make_tokenization_function(tokenizer, options):
    seq_len = options.seq_len
    def tokenize(text):
        return tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding=True,
            return_tensors='np'
        )
    return tokenize


def inputs(tokenizer_output):
    return [
        tokenizer_output['input_ids'],
        tokenizer_output['attention_mask'],
        tokenizer_output['token_type_ids']
    ]


def encode_labels(labels, label_encoder, one_hot=True):
    Y = label_encoder.transform(labels)
    if not one_hot:
        return Y
    else:
        return to_categorical(Y, num_classes=len(label_encoder.classes_))


def main(argv):
    options = argparser().parse_args(argv[1:])

    train_texts, train_labels = load_data(options.train, options)
    dev_texts, dev_labels = load_data(options.dev, options)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    num_labels = len(label_encoder.classes_)
    train_Y = encode_labels(train_labels, label_encoder)
    dev_Y = encode_labels(dev_labels, label_encoder)

    pretrained_model, tokenizer, config = load_pretrained(options)
    optimizer = get_optimizer(len(train_texts), options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options)

    tokenize = make_tokenization_function(tokenizer, options)
    train_X = tokenize(train_texts)
    dev_X = tokenize(dev_texts)

    history = model.fit(
        inputs(train_X),
        train_Y,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_data=(inputs(dev_X), dev_Y),
    )

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

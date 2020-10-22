#!/usr/bin/env python3

import sys
import math

from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC

from tensorflow_addons.metrics import F1Score

from transformers import AutoConfig, AutoTokenizer, TFAutoModel
from transformers.optimization_tf import create_optimizer

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning

from readers import READERS, get_reader
from common import timed


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
    ap.add_argument('--multiclass', default=False, action='store_true',
                    help='task has exactly one label per text')
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
    input_ids = Input(
        shape=(seq_len,), dtype='int32', name='input_ids')
    token_type_ids = Input(
        shape=(seq_len,), dtype='int32', name='token_type_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')

    pretrained_outputs = pretrained_model([
        input_ids,
        attention_mask,
        token_type_ids
    ])
    pooled_output = pretrained_outputs[1]

    # TODO consider Dropout here
    if options.multiclass:
        output = Dense(num_labels, activation='softmax')(pooled_output)
        loss = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy(name='acc')]
    else:
        output = Dense(num_labels, activation='sigmoid')(pooled_output)
        loss = BinaryCrossentropy()
        metrics = [
            F1Score(name='f1', num_classes=num_labels, average='micro',
                    threshold=0.5),
            AUC(name='auc', multi_label=True)
        ]
        
    model = Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=[output]
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


@timed
def load_data(fn, options):
    read = get_reader(options.input_format)
    texts, labels = [], []
    with open(fn) as f:
        for ln, (text, text_labels) in enumerate(read(f, fn), start=1):
            if options.multiclass and not text_labels:
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            elif options.multiclass and len(text_labels) > 1:
                raise ValueError(f'multiple labels on line {ln} in {fn}: {l}')
            texts.append(text)
            labels.append(text_labels)
    print(f'loaded {len(texts)} examples from {fn}', file=sys.stderr)
    return texts, labels


def make_tokenization_function(tokenizer, options):
    seq_len = options.seq_len
    def tokenize(text):
        tokenized = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding=True,
            return_tensors='np'
        )
        # Return dict b/c Keras (2.3.0-tf) DataAdapter doesn't apply
        # dict mapping to transformer.BatchEncoding inputs
        return {
            'input_ids': tokenized['input_ids'],
            'token_type_ids': tokenized['token_type_ids'],
            'attention_mask': tokenized['attention_mask'],
        }
    return tokenize


@timed
def prepare_classifier(num_train_examples, num_labels, options):
    pretrained_model, tokenizer, config = load_pretrained(options)
    optimizer = get_optimizer(num_train_examples, options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options)
    return model, tokenizer, optimizer


def main(argv):
    options = argparser().parse_args(argv[1:])

    train_texts, train_labels = load_data(options.train, options)
    dev_texts, dev_labels = load_data(options.dev, options)
    num_train_examples = len(train_texts)
    
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(train_labels)
    train_Y = label_encoder.transform(train_labels)
    dev_Y = label_encoder.transform(dev_labels)
    num_labels = len(label_encoder.classes_)

    classifier, tokenizer, optimizer = prepare_classifier(
        num_train_examples,
        num_labels,
        options
    )

    tokenize = make_tokenization_function(tokenizer, options)
    train_X = tokenize(train_texts)
    dev_X = tokenize(dev_texts)

    history = classifier.fit(
        train_X,
        train_Y,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_data=(dev_X, dev_Y),
    )

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

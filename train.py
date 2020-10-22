#!/usr/bin/env python3

import sys

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from transformers import AutoConfig, AutoTokenizer, TFAutoModel

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning

from readers import READERS, get_reader


# Parameter defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 128
DEFAULT_LR = 5e-5


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


def get_optimizer(options):
    return Adam(lr=options.lr)


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
    output_probs = Dense(num_labels, activation='softmax')(pooled_output)

    model = Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=[output_probs]
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
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


def main(argv):
    options = argparser().parse_args(argv[1:])

    train_texts, train_labels = load_data(options.train, options)
    dev_texts, dev_labels = load_data(options.dev, options)
    
    label_encoder = LabelEncoder()
    train_Y = label_encoder.fit_transform(train_labels)
    dev_Y = label_encoder.transform(dev_labels)
    num_labels = len(label_encoder.classes_)

    pretrained_model, tokenizer, config = load_pretrained(options)
    optimizer = get_optimizer(options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options)

    tokenize = make_tokenization_function(tokenizer, options)
    train_X = tokenize(train_texts)
    dev_X = tokenize(dev_texts)

    model.fit(
        inputs(train_X),
        train_Y,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_data=(inputs(dev_X), dev_Y)
    )

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

#!/usr/bin/env python3

import sys

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from transformers import AutoConfig, AutoTokenizer, TFAutoModel

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning


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


def load_fasttext_data(fn):
    """Load FastText format with exactly one label per line."""
    texts, labels = [], []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            label, text = l.split(None, 1)
            if not label.startswith('__label__'):
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            label = label[len('__label__'):]
            if '__label__' in text:
                warning(f'multiple labels on line {ln} in {fn}: {l}')
            texts.append(text)
            labels.append(label)
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

    train_texts, train_labels = load_fasttext_data(options.train)
    dev_texts, dev_labels = load_fasttext_data(options.dev)
    
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

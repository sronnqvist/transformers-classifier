#!/usr/bin/env python3

import sys
import math
import numpy as np
from os.path import isfile
import csv

from scipy.sparse import lil_matrix

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.config import experimental as tfconf_exp

from tensorflow_addons.metrics import F1Score

from transformers import AutoConfig, AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification
from transformers.optimization_tf import create_optimizer

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning

from readers import READERS, get_reader
from common import timed


# Parameter defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 512
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_PROPORTION = 0.1


def init_tf_memory():
    gpus = tfconf_exp.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tfconf_exp.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=None,
                    help='pretrained model name')
    ap.add_argument('--train', metavar='FILE', required=True,
                    help='training data')
    ap.add_argument('--dev', metavar='FILE', required=True,
                    help='development data')
    ap.add_argument('--test', metavar='FILE', required=False,
                    help='test data', default=None)
    ap.add_argument('--bg_train', metavar='FILE', required=False,
                    help='background corpus for training', default=None)
    ap.add_argument('--bg_sample_rate', metavar='FLOAT', type=float,
                    default=1.,
                    help='rate at which to sample from background corpus (X:1)')
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
    ap.add_argument('--output_file', default=None, metavar='FILE',
                    help='save model to file')
    ap.add_argument('--save_predictions', default=False, action='store_true',
                    help='save predictions and labels for dev set, or for test set if provided')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='load model from file')
    ap.add_argument('--log_file', default="train.log", metavar='FILE',
                    help='log parameters and performance to file')
    ap.add_argument('--test_log_file', default="test.log", metavar='FILE',
                    help='log parameters and performance on test set to file')
    return ap



def load_pretrained(options):
    name = options.model_name
    config = AutoConfig.from_pretrained(name)
    config.return_dict = True
    tokenizer = AutoTokenizer.from_pretrained(name, config=config)
    model = TFAutoModel.from_pretrained(name, config=config)
    #model = TFAutoModelForSequenceClassification.from_pretrained(name, config=config)

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
#    token_type_ids = Input(
#        shape=(seq_len,), dtype='int32', name='token_type_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')
#    inputs = [input_ids, attention_mask, token_type_ids]
    inputs = [input_ids, attention_mask]

    pretrained_outputs = pretrained_model(inputs)
    #pooled_output = pretrained_outputs[1]
    pooled_output = pretrained_outputs['last_hidden_state'][:,0,:] #CLS

    # TODO consider Dropout here
    if options.multiclass:
        output = Dense(num_labels, activation='softmax')(pooled_output)
        loss = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy(name='acc')]
    else:
        output = Dense(num_labels, activation='sigmoid')(pooled_output)
        loss = BinaryCrossentropy()
        metrics = [
            #F1Score(name='f1_th0.3', num_classes=num_labels, average='micro', threshold=0.3),
            F1Score(name='f1_th0.4', num_classes=num_labels, average='micro', threshold=0.4)#,
            #F1Score(name='f1_th0.5', num_classes=num_labels, average='micro', threshold=0.5),
            #AUC(name='auc', multi_label=True)
        ]
    #output = pretrained_outputs # test
    model = Model(
        inputs=inputs,
        outputs=[output]
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


@timed
def load_data(fn, options, max_chars=None):
    read = get_reader(options.input_format)
    texts, labels = [], []
    with open(fn) as f:
        for ln, (text, text_labels) in enumerate(read(f, fn), start=1):
            if options.multiclass and not text_labels:
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            elif options.multiclass and len(text_labels) > 1:
                raise ValueError(f'multiple labels on line {ln} in {fn}: {l}')
            texts.append(text[:max_chars])
            labels.append(text_labels)
    print(f'loaded {len(texts)} examples from {fn}', file=sys.stderr)
    return texts, labels


class DataGenerator(Sequence):
    def __init__(self, data_path, tokenize_func, options, bg_data_path=None, bg_sample_rate=1., max_chars=None, label_encoder=None):
        texts, labels = load_data(data_path, options, max_chars=max_chars)
        self.num_examples = len(texts)
        self.batch_size = options.batch_size
        #self.seq_len = options.seq_len
        self.X = tokenize_func(texts)

        if label_encoder is None:
            self.label_encoder = MultiLabelBinarizer()
            self.label_encoder.fit(labels)
        else:
            self.label_encoder = label_encoder

        self.Y = self.label_encoder.transform(labels)
        self.num_labels = len(self.label_encoder.classes_)

        if bg_data_path is not None:
            self.bg_sample_rate = bg_sample_rate
            self.bg_num_examples, self.bg_X, self.bg_Y = [], [], []
            for path in bg_data_path.split():
                bg_texts, bg_labels = load_data(path, options, max_chars=max_chars)
                self.bg_num_examples.append(len(bg_texts))
                self.bg_X.append(tokenize_func(bg_texts))

                self.bg_Y.append(self.label_encoder.transform(bg_labels))
            #self.bg_num_labels = len(self.label_encoder.classes_)
            self.bg_num_corpora = len(self.bg_num_examples)
        else:
            self.bg_sample_rate = 0
            self.bg_num_examples = [0]
            self.bg_num_corpora = 0

        self.on_epoch_end()


    def on_epoch_end(self):
        self.indexes = np.arange(self.num_examples)
        np.random.shuffle(self.indexes)

        if self.bg_sample_rate > 0:
            if hasattr(self, "bg_indexes"):
                for i, bg_indexes in enumerate(self.bg_indexes):
                    seen_bg_idxs = bg_indexes[:len(self.indexes)]
                    unseen_bg_idxs = bg_indexes[len(self.indexes):]
                    np.random.shuffle(seen_bg_idxs)
                    self.bg_indexes[i] = np.concatenate([unseen_bg_idxs, seen_bg_idxs])
            else:
                self.bg_indexes = [np.arange(x) for x in self.bg_num_examples]
                for i,_ in enumerate(self.bg_indexes):
                    np.random.shuffle(self.bg_indexes[i])

        self.index = 0


    def __len__(self):
        return int((self.num_examples//self.batch_size)*(1+self.bg_sample_rate)) * self.bg_num_corpora

    def __getitem__(self, index):
        if np.random.random() <= 1/(self.bg_sample_rate+self.bg_num_corpora):
            batch_indexes = self.indexes[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            X, Y = self.X, self.Y
        else:
            i = np.random.random_integers(0,self.bg_num_corpora-1)
            try:
                batch_indexes = self.bg_indexes[i][self.index*self.batch_size:(self.index+1)*self.batch_size]
            except IndexError:
                end = ((self.index+1)*self.batch_size) % len(self.bg_indexes[i])
                if end < self.batch_size:
                    end = self.batch_size
                    beg = 0
                else:
                    beg = end-self.batch_size
                batch_indexes = self.bg_indexes[i][beg:end]

            X, Y = self.bg_X[i], self.bg_Y[i]

        batch_X = {}
        for key in self.X:
            batch_X[key] = np.empty((self.batch_size, *X[key].shape[1:]))
            for j, idx in enumerate(batch_indexes):
                batch_X[key][j] = X[key][idx]

        batch_y = np.empty((self.batch_size, *Y.shape[1:]), dtype=int)
        for j, idx in enumerate(batch_indexes):
            batch_y[j] = Y[idx]

        return batch_X, batch_y


def make_tokenization_function(tokenizer, options):
    seq_len = options.seq_len
    @timed
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
#            'token_type_ids': tokenized['token_type_ids'],
            'attention_mask': tokenized['attention_mask'],
        }
    return tokenize


@timed
def prepare_classifier(num_train_examples, num_labels, options):
    pretrained_model, tokenizer, config = load_pretrained(options)
    optimizer = get_optimizer(num_train_examples, options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options)
    return model, tokenizer, optimizer


def optimize_threshold(model, train_X, train_Y, test_X, test_Y, options=None, epoch=None, save_pred_to=None):
    labels_prob = model.predict(train_X, verbose=1)#, batch_size=options.batch_size)

    best_f1 = 0.
    print("Optimizing threshold...\nThres.\tPrec.\tRecall\tF1")
    for threshold in np.arange(0.1, 0.9, 0.05):
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>=threshold] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(train_Y, labels_pred, average="micro")
        print("%.2f\t%.4f\t%.4f\t%.4f" % (threshold, precision, recall, f1), end="")
        if f1 > best_f1:
            print("\t*")
            best_f1 = f1
            #best_f1_epoch = epoch
            best_f1_threshold = threshold
        else:
            print()

    #print("Current F_max:", best_f1, "epoch", best_f1_epoch+1, "threshold", best_f1_threshold, '\n')
    #print("Current F_max:", best_f1, "threshold", best_f1_threshold, '\n')

    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=best_f1_threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (best_f1_threshold, epoch_str, test_precision, test_recall, test_f1))

    if save_pred_to is not None:
        if epoch is None:
            print("Saving predictions to", save_pred_to+".*.npy" % epoch)
            np.save(save_pred_to+".preds.npy", test_labels_pred.toarray())
            np.save(save_pred_to+".gold.npy", test_Y)
            #np.save(save_pred_to+".class_labels.npy", label_encoder.classes_)
        else:
            print("Saving predictions to", save_pred_to+"-epoch%d.*.npy" % epoch)
            np.save(save_pred_to+"-epoch%d.preds.npy" % epoch, test_labels_pred.toarray())
            np.save(save_pred_to+"-epoch%d.gold.npy" % epoch, test_Y)
            #np.save(save_pred_to+"-epoch%d.class_labels.npy" % epoch, label_encoder.classes_)

    return test_f1, best_f1_threshold, test_labels_pred


def test_threshold(model, test_X, test_Y, threshold=0.4, options=None, epoch=None):
    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (threshold, epoch_str, test_precision, test_recall, test_f1))
    return test_f1, threshold, test_labels_pred


def test_auc(model, test_X, test_Y):
    labels_prob = model.predict(test_X, verbose=1)
    return roc_auc_score(test_Y, labels_prob, average = 'micro')


class Logger:
    def __init__(self, filename, model, params):
        self.filename = filename
        self.model = model
        self.log = dict([('p%s'%p, v) for p, v in params.items()])

    def record(self, epoch, logs):
        for k in logs:
            self.log['_%s' % k] = logs[k]
        self.log['_Epoch'] = epoch
        self.write()

    def write(self):
        file_exists = isfile(self.filename)
        with open(self.filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, sorted(self.log.keys()))
            if not file_exists:
                print("Creating log file", self.filename, flush=True)
                writer.writeheader()
            writer.writerow(self.log)


class EvalCallback(Callback):
    def __init__(self, model, train_X, train_Y, dev_X, dev_Y, test_X=None, test_Y=None, logfile="train.log", test_logfile=None, save_pred_to=None, params={}):
        self.model = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.dev_X = dev_X
        self.dev_Y = dev_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.logger = Logger(logfile, self.model, params)
        if test_logfile is not None:
            print("Setting up test set logging to", test_logfile, flush=True)
            self.test_logger = Logger(test_logfile, self.model, params)
        if save_pred_to is not None:
            self.save_pred_to = save_pred_to
        else:
            self.save_pred_to = None

    def on_epoch_end(self, epoch, logs={}):
        print("Validation set performance:")
        logs['f1'], _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.dev_X, self.dev_Y, epoch=epoch)
        logs['rocauc'] = test_auc(self.model, self.dev_X, self.dev_Y)
        print("AUC dev:", logs['rocauc'])

        self.logger.record(epoch, logs)
        if self.test_X is not None:
            print("Test set performance:")
            test_f1, _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.test_X, self.test_Y, epoch=epoch, save_pred_to=self.save_pred_to)
            auc = test_auc(self.model, self.test_X, self.test_Y)
            print("AUC test:", auc)
            self.test_logger.record(epoch, {'f1': test_f1, 'rocauc': auc})


def main(argv):
    init_tf_memory()
    options = argparser().parse_args(argv[1:])

    ### Load data without generator
    train_texts, train_labels = load_data(options.train, options, max_chars=25000)
    dev_texts, dev_labels = load_data(options.dev, options, max_chars=25000)
    if options.test is not None:
        test_texts, test_labels = load_data(options.test, options, max_chars=25000)
    #num_train_examples = len(train_texts)

    """classifier, tokenizer, optimizer = prepare_classifier(
        num_train_examples,
        num_labels,
        options
    )"""
    ### end of data loading

    pretrained_model, tokenizer, model_config = load_pretrained(options)
    tokenize = make_tokenization_function(tokenizer, options)

    train_X = tokenize(train_texts)
    dev_X = tokenize(dev_texts)

    if options.bg_train is None:
        train_gen = DataGenerator(options.train, tokenize, options, max_chars=25000)
    else:
        train_gen = DataGenerator(options.train, tokenize, options, max_chars=25000, bg_data_path=options.bg_train, bg_sample_rate=options.bg_sample_rate)
    dev_gen = DataGenerator(options.dev, tokenize, options, max_chars=25000, label_encoder=train_gen.label_encoder)

    optimizer = get_optimizer(train_gen.num_examples*(1+train_gen.bg_sample_rate*train_gen.bg_num_corpora), options)
    classifier = build_classifier(pretrained_model, train_gen.num_labels, optimizer, options)
    """classifier, tokenizer, optimizer = prepare_classifier(
        train_gen.num_examples,
        train_gen.num_labels,
        options
    )"""

    """label_encoder = MultiLabelBinarizer()
    label_encoder.fit(train_labels)
    train_Y = label_encoder.transform(train_labels)
    dev_Y = label_encoder.transform(dev_labels)"""
    train_Y = train_gen.label_encoder.transform(train_labels)
    dev_Y = train_gen.label_encoder.transform(dev_labels)
    if options.test is not None:
        #test_Y = label_encoder.transform(test_labels)
        test_Y = train_gen.label_encoder.transform(test_labels)

    #num_labels = len(label_encoder.classes_)



    options.test = None # TODO: adapt generator to test set
    if options.test is not None:
        test_X = tokenize(test_texts)

    if options.load_model is not None:
        classifier.load_weights(options.load_model)

        print("Evaluating on dev set...")
        f1, th, dev_pred = optimize_threshold(classifier, train_X, train_Y, dev_X, dev_Y, options)
        print("AUC dev:", test_auc(classifier, dev_X, dev_Y))

        if options.test is not None:
            print("Evaluating on test set...")
            test_f1, test_th, test_pred = optimize_threshold(classifier, train_X, train_Y, test_X, test_Y, options, save_pred_to=options.load_model)
            np.save(options.load_model+".class_labels.npy", label_encoder.classes_)
            #test_f1, test_th, test_pred = optimize_threshold(classifier, train_X, train_Y, test_X, test_Y, options)
            print("AUC test:", test_auc(classifier, test_X, test_Y))

        #f1, th, dev_pred = test_threshold(classifier, dev_X, dev_Y, threshold=0.4)
        return


    callbacks = [] #[ModelCheckpoint(options.output_file+'.{epoch:02d}', save_weights_only=True)]
    if options.test is not None and options.test_log_file is not None:
        print("Initializing evaluation with dev and test set...")
        callbacks.append(EvalCallback(classifier, train_X, train_Y, dev_X, dev_Y, test_X=test_X, test_Y=test_Y,
                                    logfile=options.log_file,
                                    test_logfile=options.test_log_file,
                                    save_pred_to=options.output_file,
                                    params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
    else:
        print("Initializing evaluation with dev set...")
        callbacks.append(EvalCallback(classifier, train_X, train_Y, dev_X, dev_Y,
                                    logfile=options.log_file,
                                    params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))

    """history = classifier.fit(
        train_X,
        train_Y,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_data=(dev_X, dev_Y),
        callbacks=callbacks
    )
    """
    history = classifier.fit_generator(
        train_gen,
        #steps_per_epoch=len(train_gen),
        validation_data=dev_gen,
        initial_epoch=0,
        epochs=options.epochs,
        callbacks=callbacks
    )


    try:
        if options.output_file:
            print("Saving model to %s" % options.output_file)
            classifier.save_weights(options.output_file)
    except:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

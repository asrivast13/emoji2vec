#!/usr/bin/env python
"""Train an emoji model based on a certain set of hyperparameters

    Allows us to train one model at a time. In practice, it would probably be simpler to use
    grid_search with only one set of parameters.
"""

# External dependencies
import os
import random

import tensorflow as tf
from tensorflow.python.framework import ops
import pickle as pk

from tfrnn.hooks import LossHook, SpeedHook

# Internal dependencies
from model import Emoji2Vec
from parameter_parser import CliParser
from utils import build_kb, get_examples_from_kb, generate_embeddings, get_metrics, generate_predictions, generate_topn_based_predictions

import multiprocessing

from sklearn import metrics
import json

random.seed(123456789)

# Authorship

__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"

target_names=["wrong", "correct"]

# Execute training sequence
def __run_training():
    # Setup
    args = CliParser()
    args.print_params('EMOJI TRAINING')

    # Build knowledge base
    print('reading training data from: ' + args.data_folder)
    kb, ind2phr, ind2emoji = build_kb(args.data_folder)

    # Save the mapping from index to emoji
    pk.dump(ind2emoji, open(args.mapping_file, 'wb'))

    # Get the embeddings for each phrase in the training set
    embeddings_array = generate_embeddings(ind2phr=ind2phr, kb=kb, embeddings_file=args.embeddings_file,
                                           word2vec_file=args.word2vec_file)

    # Get examples of each example type in two sets. This is just a reprocessing of the knowledge base for efficiency,
    # so we don't have to generate the train and dev set on each train
    train_set = get_examples_from_kb(kb=kb, example_type='train')
    dev_set = get_examples_from_kb(kb=kb, example_type='dev')
    test_set = get_examples_from_kb(kb=kb, example_type='test')

    train_save_evaluate(params=args.model_params, kb=kb, train_set=train_set, dev_set=dev_set, ind2emoji=ind2emoji,
                        embeddings_array=embeddings_array, dataset_name=args.dataset, topN=args.topn, ind2phr=ind2phr, test_set=test_set)


def train_save_evaluate(params, kb, train_set, dev_set, ind2emoji, embeddings_array, dataset_name, topN=2, ind2phr=None, test_set=None):
    """Train the model on the kb, save the trained model, and evaluate it against several metrics.

    Args:
        params: Model parameters
        kb: Knowledge base
        train_set: Training set
        dev_set: Dev set
        ind2emoji: Mapping between index and emoji
        embeddings_array: Embeddings array to feed into the model
        dataset_name: Name of the dataset (i.e. unicode)

    Returns:
        Dictionary of results with various performance measures
    """
    # Reset the graph so that we don't pollute models
    ops.reset_default_graph()
    tf.reset_default_graph()

    # Generate the model path
    model_folder = params.model_folder(dataset_name=dataset_name)
    model_path = model_folder + '/model.ckpt'

    # If the minibatch is larger than the number of emojis we have, we can't can't fill train/test batches
    if params.mb > len(ind2emoji):
        print(str.format('Skipping: k={}, batch={}, epochs={}, ratio={}, dropout={}', params.out_dim,
                         params.pos_ex, params.max_epochs, params.neg_ratio, params.dropout))
        print("Can't have an mb > len(ind2emoji)")
        return "N/A"
    else:
        print(str.format('Training: k={}, batch={}, epochs={}, ratio={}, dropout={}', params.out_dim,
                         params.pos_ex, params.max_epochs, params.neg_ratio, params.dropout))

    # Create the model based on the given model parameters
    model = Emoji2Vec(model_params=params, num_emoji=kb.dim_size(0), embeddings_array=embeddings_array)

    # Map from a name to a training set
    dsets = {'train': train_set, 'dev': dev_set}
    if test_set is not None:
        dsets['test'] = test_set
    predictions = dict()
    topnpred = dict()
    results = dict()

    with tf.Session() as sess:
        saver = tf.train.Saver()

        # If we don't have a checkpoint, we must retrain it
        if os.path.exists(model_path):
            predictions = pk.load(open(model_folder + '/results.p', 'rb'))

        else:
            # For visualizing using tensorboard
            #summary_writer = tf.train.SummaryWriter(model_folder + '/board', graph=sess.graph)
            summary_writer = tf.summary.FileWriter(model_folder + '/board', graph=sess.graph)

            # Keep track of how the model is training
            hooks = [
                SpeedHook(summary_writer, 5000, params.mb),
                LossHook(summary_writer, 1000, params.mb)
            ]

            # Train the model
            model.train(kb=kb, hooks=hooks, session=sess)

            # Save a checkpoint with the trained model
            saver.save(sess, model_path)

            # Generate the gensim structures
            e2v = model.create_gensim_files(sess=sess, model_folder=model_folder, ind2emoj=ind2emoji,
                                            out_dim=params.out_dim)

            # For train and dev, make predictions and store them in our predictions dictionary
            for dset_name in dsets:
                # Make predictions, and only save the values right now
                _, pred_values, _, true_values = generate_predictions(e2v=e2v, dset=dsets[dset_name],
                                                                      phr_embeddings=embeddings_array,
                                                                      ind2emoji=ind2emoji,
                                                                      threshold=params.class_threshold)

                predictions[dset_name] = {
                    'y_true': true_values,
                    'y_pred': pred_values
                }

                if dset_name != 'train':

                    pred_file = (model_folder + '/pred_' + dset_name + '.txt') if ind2phr is not None else None
                
                    topn_pred_labels,  topn_pred_values, topn_true_labels, topn_true_values, rec = generate_topn_based_predictions(e2v=e2v, dset=dsets[dset_name],
                                                                                                                                   phr_embeddings=embeddings_array,
                                                                                                                                   ind2emoji=ind2emoji, topN=topN,
                                                                                                                                   ind2phr=ind2phr, out_file=pred_file)

                    topnpred[dset_name] = {
                        'y_true': topn_true_values,
                        'y_pred': topn_pred_values,
                        'l_true': topn_true_labels,
                        'l_pred': topn_pred_labels,
                        'rec'   : rec
                    }
                

            # Save the metrics for posterity, so we don't have to recalculate them
            pk.dump(predictions, open(model_folder + '/results.p', 'wb'))

        # For train and dev, generate and print metrics
        for dset_name in dsets:
            true_labels = [bool(x) for x in predictions[dset_name]['y_true']]
            pred_labels = [x >= params.class_threshold for x in predictions[dset_name]['y_pred']]
            true_values = predictions[dset_name]['y_true']
            pred_values = predictions[dset_name]['y_pred']

            # Calculate metrics
            acc, f1, auc = get_metrics(pred_labels, pred_values, true_labels, true_values)

            print(str.format('{}: Accuracy(>{}): {}, f1: {}, auc: {}', dset_name, params.class_threshold, acc, f1, auc))

            results[dset_name] = {
                'topn' : 0,
                'accuracy': acc,
                'f1': f1,
                'auc': auc,
                'rec': {},
            }

            if dset_name != 'train':
                # Calculate metrics
                acc, f1, auc = get_metrics(topnpred[dset_name]['l_pred'], topnpred[dset_name]['y_pred'], topnpred[dset_name]['l_true'], topnpred[dset_name]['y_true'])

                print(str.format('{}: Accuracy(top{}): {}, f1: {}, auc: {}', dset_name, topN, acc, f1, auc))

                rec = topnpred[dset_name]['rec']
                results[dset_name]['topn'] = acc
                results[dset_name]['rec']  = rec
                

                print(str.format('{}: Raw(top{}): Tot:{}, NP:{}, NN:{}, TP:{}, TN:{}, FP:{}, FN:{}', dset_name, topN, rec['tot'], rec['np'], rec['nn'], rec['tp'], rec['tn'], rec['fp'], rec['fn'])) 
            
                #print(str.format('\n{} Set Classification Report:\n {}', dset_name, metrics.classification_report(topnpred[dset_name]['l_true'], topnpred[dset_name]['l_pred'])))
                #print("\n\n")

    with open(model_folder + '/results.json', 'w') as fp:
        fp.write(json.dumps(results))

    return results['dev']


if __name__ == '__main__':

    multiprocessing.set_start_method('forkserver', force=True)
    multiprocessing.freeze_support()
    __run_training()

import pandas as pd
import numpy as np

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


def remove_seq_padding(X, y_true, y_pred, pad=None):
    new_true, new_pred = [], []
    for x_sent, true_sent, pred_sent in zip(X, y_true, y_pred):
        if pad is None:
            pad = len(x_sent)
        # Remove padding elements from the true and predicted sequences
        true_sent = true_sent[:pad]
        pred_sent = pred_sent[:pad]

        new_true.append(true_sent)
        new_pred.append(pred_sent)
    return np.array(new_true), np.array(new_pred)


def from_encode_to_literal_labels(y_true, y_pred, idx2tag):
    '''Transform sequences of encoded labels in sequences of string labels'''
    let_y_true = list()
    let_y_pred = list()
    for sent_idx in range(len(y_true)):
        let_sent_true = []
        let_sent_pred = []
        for token_idx in range(len(y_true[sent_idx])):
            let_sent_true.append(idx2tag[y_true[sent_idx][token_idx]])
            let_sent_pred.append(idx2tag[y_pred[sent_idx][token_idx]])
        let_y_true.append(let_sent_true)
        let_y_pred.append(let_sent_pred)
    
    return let_y_true, let_y_pred
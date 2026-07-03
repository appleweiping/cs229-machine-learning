"""PS2 Problem 6: SMS spam classification with Naive Bayes and an RBF SVM.

Messages are tokenised on whitespace, lowercased, and turned into word-count
vectors over a dictionary of words appearing in >= 5 training messages.  A
multinomial Naive Bayes model with Laplace smoothing is fitted in closed form;
an RBF-kernel SVM is tuned over a small grid of radii for comparison.
"""
import numpy as np

import util
import svm


def get_words(message):
    """Lowercase and split a message on whitespace."""
    return message.lower().split()


def create_dictionary(messages):
    """Map words occurring in >= 5 messages to contiguous integer indices."""
    doc_counts = {}
    for message in messages:
        for word in set(get_words(message)):
            doc_counts[word] = doc_counts.get(word, 0) + 1
    return {word: i for i, word in enumerate(w for w, c in doc_counts.items() if c >= 5)}


def transform_text(messages, word_dictionary):
    """Return an (n_messages, vocab) matrix of per-message word counts."""
    matrix = np.zeros((len(messages), len(word_dictionary)))
    for row, message in enumerate(messages):
        for word in get_words(message):
            col = word_dictionary.get(word)
            if col is not None:
                matrix[row, col] += 1
    return matrix


def fit_naive_bayes_model(matrix, labels):
    """Fit multinomial Naive Bayes with Laplace smoothing.

    Returns log token probabilities per class and the log class prior, so that
    prediction is a single matrix multiply.
    """
    vocab = matrix.shape[1]
    counts_1 = matrix[labels == 1].sum(axis=0)
    counts_0 = matrix[labels == 0].sum(axis=0)

    log_phi_1 = np.log((counts_1 + 1) / (counts_1.sum() + vocab))
    log_phi_0 = np.log((counts_0 + 1) / (counts_0.sum() + vocab))
    log_prior = np.log(np.mean(labels))
    log_prior_neg = np.log(1 - np.mean(labels))
    return log_phi_1, log_phi_0, log_prior, log_prior_neg


def predict_from_naive_bayes_model(model, matrix):
    """Predict {0,1} by comparing class log-joint scores."""
    log_phi_1, log_phi_0, log_prior, log_prior_neg = model
    score_1 = matrix @ log_phi_1 + log_prior
    score_0 = matrix @ log_phi_0 + log_prior_neg
    return (score_1 > score_0).astype(int)


def get_top_five_naive_bayes_words(model, dictionary):
    """Top-5 words by log p(word|spam) / p(word|ham) (metric from part 6c)."""
    log_phi_1, log_phi_0, _, _ = model
    llr = log_phi_1 - log_phi_0
    idx_to_word = {i: w for w, i in dictionary.items()}
    top = np.argsort(-llr)[:5]
    return [idx_to_word[i] for i in top]


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels,
                            radius_to_consider):
    """Select the RBF radius with highest validation accuracy."""
    best_radius, best_acc = None, -1.0
    for radius in radius_to_consider:
        pred = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        acc = np.mean(pred == val_labels)
        print(f'[p06] SVM radius={radius}: validation accuracy = {acc:.4f}')
        if acc > best_acc:
            best_acc, best_radius = acc, radius
    return best_radius


def main(data_dir='../../data/ps2'):
    import os
    os.makedirs('output', exist_ok=True)

    train_messages, train_labels = util.load_spam_dataset(f'{data_dir}/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset(f'{data_dir}/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset(f'{data_dir}/ds6_test.tsv')

    dictionary = create_dictionary(train_messages)
    print(f'[p06] dictionary size = {len(dictionary)}')
    util.write_json('output/p06_dictionary.json', dictionary)

    train_matrix = transform_text(train_messages, dictionary)
    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    nb_model = fit_naive_bayes_model(train_matrix, train_labels)
    nb_pred = predict_from_naive_bayes_model(nb_model, test_matrix)
    nb_acc = np.mean(nb_pred == test_labels)
    print(f'[p06] Naive Bayes test accuracy = {nb_acc:.4f}')

    top_5 = get_top_five_naive_bayes_words(nb_model, dictionary)
    print(f'[p06] top 5 spam-indicative words = {top_5}')
    util.write_json('output/p06_top_indicative_words.json', top_5)

    radius = compute_best_svm_radius(train_matrix, train_labels,
                                     val_matrix, val_labels, [0.01, 0.1, 1, 10])
    util.write_json('output/p06_optimal_radius.json', radius)
    svm_pred = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, radius)
    svm_acc = np.mean(svm_pred == test_labels)
    print(f'[p06] optimal SVM radius = {radius}; test accuracy = {svm_acc:.4f}')

    return dict(vocab=len(dictionary), nb_acc=nb_acc, top_5=top_5,
                radius=radius, svm_acc=svm_acc)


if __name__ == '__main__':
    main()

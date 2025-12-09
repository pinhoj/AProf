#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json
import numpy as np
import utils


def relu(matrix):
    return np.maximum(0, matrix)


def drelu(matrix):
    return matrix > 0


def softmax(matrix):
    a = np.exp(matrix)
    return a / np.sum(a)


class MLPerceptron:
    def __init__(self, n_classes, n_hidden, n_features):
        self.W1 = np.random.normal(0.1, 0.01, (n_hidden, n_features))
        self.B1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.normal(0.1, 0.01, (n_classes, n_hidden))
        self.B2 = np.zeros((n_classes, 1))
        self.eta = 0.001


    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)


    def update_weight(self, x_i, y_i, z1, hidden, output):
        """
        x_i (n_features,): a single training example
        y_i (scalar): the gold label for that example
        """
        # y is a 1-hot vector
        # loss = output - y
        # so loss is output with y_i index -1
        output[y_i] -= 1
        dw2 = output @ hidden.T
        db2 = output
        dh1 = self.W2.T @ output
        loss1 = np.multiply(dh1, drelu(z1))
        self.W2 -= self.eta * dw2
        self.B2 -= self.eta * db2 
        
        dw1 = loss1 @ x_i.T
        db1 = loss1
        self.W1 -= self.eta * dw1
        self.B1 -= self.eta * db1


    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        # Todo: Q1 1(a)
        X = X.reshape(len(X), -1, 1)
        for (x, label) in zip(X,y):
            z1 = self.W1 @ x + self.B1
            hidden = relu(z1)
            output = softmax(self.W2 @ hidden + self.B2) 
            self.update_weight(x, label, z1, hidden, output)
        

    def forward(self, X):
        z1 = self.W1 @ X.T + self.B1
        hidden = relu(z1)
        output = self.W2 @ hidden + self.B2
        ## no need for softmax because argmax stays the same
        return output
    

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        # Todo: Q1 1(a)
        return np.argmax(self.forward(X), axis=0)
        

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        preds = self.predict(X)
        return np.mean(preds == y)
        

def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]
    n_hidden = 100
    # initialize the model
    model = MLPerceptron(n_classes, n_hidden, n_feats)

    epochs = np.arange(1, args.epochs + 1)

    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        model.train_epoch(X_train, y_train)
        # model.load(args.save_path)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        # Todo: Q1(a)
        # Decide whether to save the model to args.save_path based on its
        # validation score
        if valid_acc >= best_valid:
            print("time {:.0f} minutes and {:.2f} seconds".format((time.time() - start) // 60,(time.time() - start) % 60))
            best_epoch = i
            print("new best model found! saving progress...")
            best_valid = valid_acc
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = MLPerceptron.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-mlp-accs.pdf")
    parser.add_argument("--scores", default="Q1-mlp-scores.json")
    args = parser.parse_args()
    main(args)

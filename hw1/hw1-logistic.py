#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json
import numpy as np
import utils


def softmax(matrix):
    a = np.exp(matrix)
    return a / np.sum(a)


def batch_softmax(matrix):
    a = np.exp(matrix)
    return a / np.sum(a, axis = 0, keepdims=True)


class LogisticRegressor:
    def __init__(self, n_classes, n_features, eta, l2):
        self.W = np.zeros((n_classes, n_features))
        self.eta = eta
        self.l2 = l2


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


    def update_weight(self, x_i, y_i, z_i):
        """
        x_i (n_features,): a single training example
        y_i (scalar): the gold label for that example
        """
        # Todo: Q1 1(a)
        z_i[y_i] -= 1
        grad = np.outer(z_i, x_i)
            
        self.W -= self.eta * (self.l2 * self.W + grad)


    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        for x, yi in zip(X, y):
            # Compute probability vector
            prob = softmax(self.W @ x)
            self.update_weight(x, yi, prob)  
        

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        ## softmax is not needed because argmax will be the same
        predictions =  np.argmax(self.W @ X.T, axis = 0)
    
        return predictions

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        preds = self.predict(X)
        return np.mean(np.equal(preds,y))
        


def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]
    
    ##2.1
    # initialize the model
    model = LogisticRegressor(n_classes, n_feats, 0.0001, 0.00001)

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
    best_model = LogisticRegressor.load(args.save_path)
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
    

def two_two(args):
    ##2.2
    ##idea: do PCA on inputs
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, _ = data["train"]
    

    X_centered = X_train - X_train.mean(axis = 0)


    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
   

    n_samples = X_train.shape[0]
    explained_variance = (S**2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Choose k components to retain e.g., 95% variance
    k = np.searchsorted(cumulative_variance, 0.95) + 1
    components = Vt[:k]
    return components

    

def grid_search(args, components):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    x_pca = X_train @ components.T
    x_valid_pca = X_valid @ components.T
    x_test_pca = X_test @ components.T

    n_components = x_pca.shape[1]
    complete_best_acc = 0
    model_i = 0
    model_desc = []
    best_model = "grid/model_6_best_valid"
    bestmodel_idx = 0
    bestmodel_test = X_test

    train = [x_pca, X_train]
    valid = [x_valid_pca, X_valid]
    test = [x_test_pca, X_test]

    for eta in [0.0001, 0.001, 0.01]:
        break
        for l2 in [0.00001, 0.001]:
    

            models = [LogisticRegressor(n_classes, n_components, eta, l2), LogisticRegressor(n_classes, n_feats, eta, l2)]
            
            for x_train, x_valid, x_test, model in zip(train, valid, test, models): 
                model_i += 1
                epochs = np.arange(1, 20 + 1)

                valid_accs = []
                train_accs = []

                start = time.time()

                best_valid = 0.0
                for i in epochs:
                    print('Training epoch {}'.format(i))
                    train_order = np.random.permutation(x_train.shape[0])
                    curr_train = x_train[train_order]
                    curr_y_train = y_train[train_order]

                    model.train_epoch(curr_train, curr_y_train)

                    train_acc = model.evaluate(curr_train, curr_y_train)
                    valid_acc = model.evaluate(x_valid, y_valid)

                    train_accs.append(train_acc)
                    valid_accs.append(valid_acc)

                    print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

                    # Todo: Q1(a)
                    # Decide whether to save the model to args.save_path based on its
                    # validation score
                    if valid_acc >= best_valid:
                        best_valid = valid_acc
                        model.save(f"grid/model_{model_i}_best_valid")
                    if valid_acc >= complete_best_acc:
                        print("new best model found! saving progress...")
                        complete_best_acc = valid_acc
                        best_model = f"grid/model_{model_i}_best_valid"
                        bestmodel_idx = i
                        bestmodel_test = x_test


                elapsed_time = time.time() - start
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                print('Training took {} minutes and {} seconds'.format(minutes, seconds))
                model_desc.append(f"{model_i}: eta = {eta}; l2 = {l2}; pca :{x_train.shape[1]!=785}; best val acc: {best_valid:.2f}; training time: {minutes}:{seconds}")
    
    
    for d in model_desc:
        print(d)
    print("Reloading best checkpoint")
    best_model = LogisticRegressor.load(best_model)
    test_acc = best_model.evaluate(bestmodel_test, y_test)
    # print(f'Best Model: {model_desc[bestmodel_idx]}')
    print('Best model test acc: {:.4f}'.format(test_acc))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-logistic-accs.pdf")
    parser.add_argument("--scores", default="Q1-logistic-scores.json")
    args = parser.parse_args()
    # main(args)
    x_pca = two_two(args)
    grid_search(args, x_pca)

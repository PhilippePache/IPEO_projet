import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.losses import CrossEntropy # added
from src.activations import Sigmoid, ReLU
from src.activations import Identity, Softmax # added
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
from src.utils import get_n_classes, label_to_onehot, onehot_to_label # added
import os
import time # added
import matplotlib.pyplot as plt # added

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    mean_val = np.mean(train_features, axis=0, keepdims=True)
    std_val = np.std(train_features, axis=0, keepdims=True)
    train_features = normalize_fn(train_features, mean_val, std_val)
    test_features = normalize_fn(test_features, mean_val, std_val)

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        assert args.method != "kmeans", f"You should consider the task clustering for K-means method."
        if args.method == "mlp":
            ## Perform K-fold cross validation to find the best hyperparameters
            # Obtain the range
            hyper_list = range(1,10)
            lr_list = [np.round(0.1*i, 2) for i in hyper_list] # visualization: plot(lr_list, acc_list)
            
            # Initialize MSE / accuracies list for each tested hyperparameter for further plotting
            perf_list = []
            for lr in lr_list: # Iterate over the learning rates
                print(f"Testing learning rate: {lr}")
                N = train_features.shape[0] # Number of data samples
                kfolds = args.kfold # Number of validation folds

                # Initialize MSE / accuracies list for computing the mean per tested hyperparameter 
                performance = []
                for fold_ind in range(kfolds): # Iterate over the K-fold validation sets
                    ## Split the data into training and validation folds:
                    
                    # Obtain all the indices of the training dataset
                    all_ind = np.arange(N)
                    split_size = N // kfolds
                    
                    # Obtain indices of the validation and training examples
                    val_ind = all_ind[fold_ind * split_size : (fold_ind + 1) * split_size]
                    train_ind = np.setdiff1d(all_ind, val_ind, assume_unique=True)
                    
                    # Split the data
                    X_train_fold = train_features[train_ind, :]
                    X_val_fold = train_features[val_ind, :]
                    if args.task == "regression":
                        Y_train_fold = train_labels_reg.copy()[train_ind]
                        Y_val_fold = train_labels_reg.copy()[val_ind]
                    else:
                        Y_train_fold = train_labels_classif.copy()[train_ind]
                        Y_val_fold = train_labels_classif.copy()[val_ind]

                    # Other preprocessing on data
                    X_train_fold = append_bias_term(X_train_fold)
                    X_val_fold = append_bias_term(X_val_fold)

                    ## Prepare the method:

                    # Set MLP dimensions based on feature dimensions and task
                    input_dim = X_train_fold.shape[1]
                    output_dim = 1 if args.task == "regression" else get_n_classes(Y_train_fold)
                    dimensions = (input_dim, args.hidden_dim, output_dim)

                    # Set activations based on task and user input
                    if args.activation == "ReLU":
                        activation_fn = ReLU
                    elif args.activation == "Sigmoid":
                        activation_fn = Sigmoid
                    else:
                        raise ValueError(f"Unknown activation function: {args.activation}")
                    activations = (activation_fn, Identity) if args.task == "regression" else (activation_fn, activation_fn, Softmax)

                    # Initialize the MLP method object
                    method_obj = MLP(dimensions=dimensions, activations=activations)
                    
                    # Train and evaluate the method
                    if args.task == "regression":
                        loss_fn = MSE
                        method_obj.fit(X_train_fold, Y_train_fold.reshape(-1, 1), loss=loss_fn, epochs=args.max_iters, batch_size=args.batch_size, learning_rate=lr)
                        Y_val_fold_pred = method_obj.predict(X_val_fold).flatten()
                        perf = mse_fn(Y_val_fold_pred, Y_val_fold)
                    else:
                        loss_fn = CrossEntropy
                        method_obj.fit(X_train_fold, label_to_onehot(Y_train_fold), loss=loss_fn, epochs=args.max_iters, batch_size=args.batch_size, learning_rate=lr)
                        Y_val_fold_pred = onehot_to_label(method_obj.predict(X_val_fold))
                        perf = accuracy_fn(Y_val_fold_pred, Y_val_fold)

                    # Compute performance for this hyperparameter and this validation set
                    
                    performance.append(perf)
                
                # Compute average performance for this hyperparameter over all validation sets
                ave_perf = np.mean(performance)
                perf_list.append(ave_perf)

            # Select the best learning rate for MLP
            lr = lr_list[np.argmin(perf_list) if args.task == "regression" else np.argmax(perf_list)]
            print(f"Best lr: {lr}")
            print(f"Performances ({args.task}): {perf_list}")
    else: # Recall: if args.test ==> Here: no K-fold cross-validation performed
        if args.method == "mlp":
            lr = args.lr

    ### WRITE YOUR CODE HERE to do any other data processing

    if args.task == "clustering":
        assert args.method != "mlp", f"You should use MLP as a classification / regression method"
        
        start_time = time.time()

        interval = range(30)
        Ks = [(i+1)*get_n_classes(train_labels_classif) for i in interval]
        distances = []
        for k in Ks:
            method_obj = KMeans(K=k, max_iters=args.max_iters)
            method_obj.fit(train_features, train_labels_classif)
            dist = method_obj.compute_distance(train_features, method_obj.centers)
            distances.append(np.mean(np.min(dist, axis=1)))

        end_time = time.time()

        # Plot the elbow curve
        plt.figure(figsize=(8, 3))
        plt.plot(Ks, distances, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Average Distance\nto Nearest Cluster Center')
        plt.xticks(Ks)
        plt.grid()
        plt.show()

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        train_features = append_bias_term(train_features)
        test_features = append_bias_term(test_features)

        # Set MLP dimensions based on feature dimensions and task
        input_dim = train_features.shape[1]
        output_dim = 1 if args.task == "regression" else get_n_classes(train_labels_classif)
        dimensions = (input_dim, args.hidden_dim, output_dim)

        # Set activations based on task and user input
        if args.activation == "ReLU":
            activation_fn = ReLU
        elif args.activation == "Sigmoid":
            activation_fn = Sigmoid
        else:
            raise ValueError(f"Unknown activation function: {args.activation}")
        activations = (activation_fn, Identity) if args.task == "regression" else (activation_fn, activation_fn, Softmax)

        # Initialize the MLP method object
        method_obj = MLP(dimensions=dimensions, activations=activations)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method
    
    if args.task == "classification":

        start_time = time.time()

        if args.method == "mlp": 
            # Fit the method on training data      
            print(f"Training MLP with learning rate: {lr}")
            method_obj.fit(train_features, label_to_onehot(train_labels_classif), loss=CrossEntropy,
                           epochs=args.max_iters, batch_size=args.batch_size, learning_rate=lr)
            # Predict on training split
            preds_train = onehot_to_label(method_obj.predict(train_features))
            # Predict on test split
            preds = onehot_to_label(method_obj.predict(test_features))       

        elif args.method == "kmeans":
            # Fit the method on training data and predict on training split
            preds_train = method_obj.fit(train_features, train_labels_classif)
            # Predict on test split
            preds = method_obj.predict(test_features)

        end_time = time.time()

        # Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"
        
        print(f"Training MLP with learning rate: {lr}")
        start_time = time.time()

        # Fit the method on training data
        method_obj.fit(train_features, train_labels_reg.reshape(-1, 1), loss=MSE, epochs=args.max_iters, batch_size=args.batch_size, learning_rate=lr)
        # Predict on training split
        preds_train = method_obj.predict(train_features).flatten()
        # Predict on test split
        preds = method_obj.predict(test_features).flatten()

        end_time = time.time()

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")    
    else:
        raise ValueError(f"Unknown task for test set: {args.task}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    # Print the runtime
    print(f"Runtime for {args.method}: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=8,
        help="number of hidden units for neural network methods",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        help="activation function for hidden layers (ReLU, Sigmoid)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size for neural network methods",
    )

    parser.add_argument(
        "--kfold",
        type=int,
        default=4,
        help="number of folds for cross-validation",
    )

    args = parser.parse_args()
    main(args)

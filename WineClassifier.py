import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=3, split_criterion='gini'):
        ''' constructor '''
        # initialize the root of the tree
        self.root = None
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.split_criterion = split_criterion

    def fit(self, X, y, param_tuning=False):
        ''' function to train the tree '''
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        if param_tuning:
            # Parameter tuning section
            best_params = self.tune_parameters(X, y)
            self.min_samples_split = best_params['min_samples_split']
            self.max_depth = best_params['max_depth']
            self.split_criterion = best_params['split_criterion']

        self.root = self.build_tree(dataset)

    def tune_parameters(self, X, y, min_samples_split_lst=[128, 64, 32, 16, 8, 4, 2], max_depth_lst=[32, 16, 8, 4, 2]):
        ''' function to perform parameter tuning '''
        best_accuracy = 0
        best_params = {'min_samples_split': None, 'max_depth': None, 'split_criterion': None}

        for min_samples_split in min_samples_split_lst:
            for max_depth in max_depth_lst:
                for split_criterion in ['gini', 'entropy']:
                    classifier = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                                        max_depth=max_depth,
                                                        split_criterion=split_criterion)
                    classifier.fit(X, y)
                    accuracy = np.mean(classifier.predict(X) == y)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params['min_samples_split'] = min_samples_split
                        best_params['max_depth'] = max_depth
                        best_params['split_criterion'] = split_criterion

        print(f"Best parameters: min_samples_split = {best_params['min_samples_split']}, "
              f"max_depth = {best_params['max_depth']}, split_criterion = {best_params['split_criterion']}")

        return best_params

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get the current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if children are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, self.split_criterion)
                    # curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return the best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="gini"):
        ''' function to compute information gain '''
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        ''' function to compute entropy '''
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        ''' function to compute gini index '''
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=""):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print("Leaf Node: Class =", tree.value)
        else:
            print(f"Decision Node: X_{tree.feature_index} <= {tree.threshold} (Information Gain: {tree.info_gain})")

            print(indent + "Left:")
            self.print_tree(tree.left, indent + "  ")

            print(indent + "Right:")
            self.print_tree(tree.right, indent + "  ")

    def predict(self, X):
        ''' function to predict new dataset '''
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=None):
        ''' function to split the dataset into training and testing sets '''
        if random_state:
            np.random.seed(random_state)

        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        test_size = int(test_size * num_samples)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return X_train, X_test, y_train, y_test


# Load the dataset
df = pd.read_csv('Wines.csv', header=None)
X = df.iloc[:, 1:].values  # Features (excluding the first column)
y = df.iloc[:, 0].values  # Labels (the first column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = DecisionTreeClassifier.train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment with different parameter tuning configurations

accuracy_arc = []
best_params_arc = {'min_samples_split': None, 'max_depth': None, 'accuracy': 0}

for mss in [2, 8, 32, 64, 128]:
    for ds in [2, 4, 8, 16, 32]:
        # Create the DTC
        classifier = DecisionTreeClassifier(min_samples_split=mss, max_depth=ds, split_criterion='gini')
        # Train the DTC
        classifier.fit(X_train, y_train, param_tuning=False)
        # Make predictions on the test set
        predictions = classifier.predict(X_test)
        # Evaluate the classifier on the test set
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy on testing set (mss={mss}, ds={ds}): {accuracy * 100:.2f}%")

        # Update best parameters if current accuracy is higher
        if accuracy > best_params_arc['accuracy']:
            best_params_arc['min_samples_split'] = mss
            best_params_arc['max_depth'] = ds
            best_params_arc['accuracy'] = accuracy

        accuracy_arc.append(accuracy)

# Print the best parameters
print(f"\nBest Parameters: min_samples_split = {best_params_arc['min_samples_split']}, "
      f"max_depth = {best_params_arc['max_depth']}, "
      f"accuracy = {best_params_arc['accuracy'] * 100:.2f}%\n")

# Instantiate and train the DecisionTreeClassifier for the best configuration
best_classifier = DecisionTreeClassifier(min_samples_split=best_params_arc['min_samples_split'],
                                          max_depth=best_params_arc['max_depth'],
                                          split_criterion='gini')
best_classifier.fit(X_train, y_train, param_tuning=False)

# Display the decision tree for the best configuration
print("\nDECISION TREE CONSTRUCTION: \n")
best_classifier.print_tree()

################################################
# Using Scikit Learn for results comparison.
# Feel free to uncomment this section to check the output given by this implementation.
################################################
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
#
# print("\nSimulating decision tree using sklearn...\n")
#
# # Load the data
# df = pd.read_csv('Wines.csv')
# y = df.iloc[:, 0].values   # Labels (the first column)
# X = df.iloc[:, 1:].values  # Features (excluding the first column)
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# accuracy_sklearn = []
# for mss in [12, 8, 32, 64, 128]:
#     for ds in [2, 4, 8, 16, 32]:
#         # Create the DTC
#         classifier = DecisionTreeClassifier(min_samples_split=mss, max_depth=ds)
#         # Train the DTC
#         classifier.fit(X_train, y_train)
#         # Make predictions on the test set
#         predictions = classifier.predict(X_test)
#         # Evaluate the classifier on the test set
#         accuracy = accuracy_score(y_test, predictions)
#         print(f'Accuracy of the Decision Tree Classifier: {accuracy * 100:.2f}%')
#
#         accuracy_sklearn.append(accuracy)
#
# print("ARC: ", accuracy_arc)
# print("sklearn: ", accuracy_sklearn)
#print("error: ", np.abs(np.asarray(accuracy_arc) - np.asarray(accuracy_sklearn)))
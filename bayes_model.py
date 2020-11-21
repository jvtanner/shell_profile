import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import sys

def debug(input):
    if False:
        print((input))


def str2bin(features_csvfile):
    """
    Turns the values of DF to binary.
    :param features_csvfile:
    :return:
    """
    # Read in the file
    df_features = pd.read_csv(features_csvfile, header=0)
    # Turn 'y' and 'site' to 1, 'n' and 'nonsite' to 0
    df_features = df_features.replace(["y"], 1)
    df_features = df_features.replace(["n"], 0)
    df_features['SITE'] = df_features['SITE'].replace(['site', 'nonsite'], [1, 0])
    debug('total matrix: {}'.format(df_features))
    return df_features


def confuse_matrix(y_test, prediction):
    """
    Create confusion matrix values.
    :param y_test:
    :param prediction:
    :return: str: True negatives, False negatives, False Positives, True Positives
    """
    tn, fp, fn, tp = sk.metrics.confusion_matrix(y_test.values.ravel(), prediction).ravel()
    debug('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))
    return tn, fp, fn, tp


class BayesClassify:
    """Class which can be trained, tested, and used for binary predictions"""

    def __init__(self):
        pass

    def train_test_sets(self, df_features):
        """
        Divide data into 80% train and 20% test.
        First split dataframe into feature vectors and prediction vector.
        :param df_features: DataFrame
        :return:
        """
        # Remove the labels from the raw data
        self.features = df_features.iloc[:, :-1]
        debug('features: {}'.format(self.features))
        # Isolate the labels
        self.labels = df_features.iloc[:, -1:]
        debug('labels: {}'.format(self.labels))
        # Allocate the data accordingly
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.features, self.labels, test_size=0.2, random_state=1115)

    def bernoulli_train(self, df_features):
        """
        Train the model using the training data.
        :param df_features: DataFrame
        :return:
        """
        self.train_test_sets(df_features)
        self.bayes_clf = BernoulliNB()
        self.bayes_clf.fit(self.X_train, self.y_train.values.ravel())

    def predict(self):
        """
        Use model to make predictions
        :return: str
        """
        prediction = self.bayes_clf.predict(self.X_test)

        debug('y-test before: {}'.format(self.y_test))
        debug('y-test after: {}'.format(self.y_test.values.ravel()))
        debug('prediction: {}'.format(prediction))

        return prediction


class FeatureIM:
    """Perform a feature evaluation on the data"""

    def __init__(self, df_features, labels):

        # Store column labels
        self.column_labels = list(df_features.columns.values)[:-1]
        debug('column labels: {}'.format(self.column_labels))

        # Drop the 'site'/'nonsite' column
        df_features.drop(df_features.columns[len(df_features.columns)-1], axis=1, inplace=True)
        self.features_of_interest = mutual_info_classif(df_features, labels.values.ravel(), discrete_features=True)

        # Put the features of interest into a DataFrame
        self.raw_foi = pd.DataFrame(self.features_of_interest)
        debug('raw foi: {}'.format(self.raw_foi))

    def raw_feat_grouping(self):
        """
        Sort by the features most correlated with a 'site'
        :return:
        """
        features_of_interest = pd.DataFrame(self.features_of_interest, index=self.column_labels, columns=['MI'])
        debug('features of interest: \n{}'.format(features_of_interest))

        sorted_features = features_of_interest.sort_values(by=['MI'])
        debug('sorted foi: \n{}'.format(sorted_features))
        return sorted_features

    def shell_grouping(self, num_shell):
        """
        Group by the shells.
        :param num_shell:
        :return:
        """
        shells = {}
        for i in range(num_shell):
            indexes = [n+i for n in range(len(self.column_labels)) if n % num_shell == 0]
            debug('indexes for shell{}: {}\n'.format(i, indexes))
            sum_MI = self.raw_foi.iloc[indexes]
            debug('sum for shell{}: {}\n'.format(i, sum_MI.sum()))
            shells['shell_{}'.format(i)] = sum_MI.sum()
        debug('Shell: {}'.format(shells))
        return shells

    def AA_grouping(self, num_AA, num_shell):
        """
        Group by amino acids.
        :param num_AA:
        :param num_shell:
        :return:
        """
        aa_sort = {}
        for i in range(num_AA):
            aa_MI = self.raw_foi.iloc[(i*num_shell):(i*num_shell+num_shell)]
            aa_sort[self.column_labels[i*num_shell]] = aa_MI.sum()
        debug('AA\'s: {}'.format(aa_sort))
        return aa_sort

def main():
    """
    Create a Bayes Classifier using the data in features.csv to predict
    whether or not a given position is a 'site' or 'nonsite'.
    Then, makes certain groupings with FeatureIM based on the ability
    to predict: raw, shells, and amino acids
    :return:
    """
    features_csvfile = sys.argv[1]
    shells = 5
    aa = 20

    # Features to binary
    df_features = str2bin(features_csvfile)

    # Create classifier
    bc = BayesClassify()
    bc.bernoulli_train(df_features)

    # Make prediction
    prediction = bc.predict()

    # Calculate confusion matrix
    tn, fp, fn, tp = confuse_matrix(bc.y_test, prediction)

    # Prepare to analyze features
    fim = FeatureIM(df_features, bc.labels)

    shell_sort = fim.shell_grouping(shells)
    aa_sort = fim.AA_grouping(aa, shells)

    print('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))
    print("Sorted Features: {}".format(fim.raw_feat_grouping()))
    print("Features sorted by shell: {}".format(shell_sort))
    print("Features sorted by AA: {}".format(aa_sort))


if __name__ == "__main__":
    main()

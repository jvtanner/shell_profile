# bayes classifier

The features.csv file contains a list of protein binding sites 
and non-binding sites. The features are represented by 'y' and 'n'
which indicate the presence or absence, respectively, of amino acids
within various shell distances away from the center of site.

bayes_model.py creates and trains a Bayesian predictor, outputs a 
confusion matrix to access accuracy. Also is able to sort the 
data by amino acids and shell space to see which features are 
most indicative of protein binding site identity.

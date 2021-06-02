Description of all the files this project includes:

1. create_df: in this file we create our design matrix by reading the given codes files.
The matrix has 2 columns: the samples and their labels. The data is converted to data frame format.

2. add_features: in this file we created the features for our samples, using the TF-IDF method which creates a vector
for each sample, by the words in it - according their frequency in each project.

3. fit_model: in this file we create our model. first, splitting our data to train and test and then fitting the model.
this file, also include the predict method that we used for calculating the accuracy.

4. model: this file includes our fitting model which (hopefully) predicts correctly.

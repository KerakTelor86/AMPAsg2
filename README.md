# 0AL5430 - Adaptive Media Processing

Richard Alison  
202320694  
University of Tsukuba

## Assignment 2
### Fisher’s Iris Classification Problem
This is a classical benchmark problem in pattern recognition which asks to classify the 3 species of plants (iris) from the characteristics observed in their flowers.
Reference : Fisher, R.A. (1936). "The Use of Multiple Measurements in Taxonomic Problems". Annals of Eugenics 7: 179–188.

### Classes (3 classes)
1. Iris Setosa 2. Iris Versicolour 3. Iris Virginica
Features
1. Petal length (cm), 2. Petal width (cm),
3. Sepal length (cm), 4. Sepal width (cm)
Data
50 examples/class. 150 examples in total.

### Problem
Download the Fisher’s iris data from the Machine Learning Repository at UC Irvine
http://archive.ics.uci.edu/ml/datasets/Iris
In the file “iris.data”, each row represents a measurement from an iris flower. Each row include the 4 features and the species (class).

Divide the data of each class into half. Use one half for training (training set), and the other half for testing (test set). Build classifiers for this data by the following two methods A and B.

A. k-nearest neighbor classifier.

B. Fisher’s linear discriminator. Prepare 3 classifiers for 2-class problems classifying one class against the other two. You may also be interested to seek information about the Multi-class Linear Discriminant Analysis in the literature.

The classifiers should be trained using the training set, and evaluated using the test set. Repeat (1) random selection of training set, (2) training, and (3) testing for 20 times. Evaluate the mean correct classification rates and their variances for both methods. Discuss what causes the differences.

You may use the programming language of your choice. Program the core classifier parts by reviewing the course materials and see if it really works as explained in the lecture. Do not use ready-made tools (e.g. the “LinearDiscriminantAnalysis” package of scikit-learn). Do not copy-paste Fisher-LDA programs floating around on the net. This is a must. In the report, include your source codes.


### Results

    KNN (K=7) Accuracy
    Mean    : 0.956000
    Variance: 0.000446

    Fisher's LDA Accuracy
    Mean    : 0.787333
    Variance: 0.003617

In the case of this problem, Fisher's LDA does not work very well when used with the described method of creating 3 separate classifiers. This may be due factors such as the difficulty in comparing the results of the classifiers.

Specifically in this work, each classifier was made to output the relative likelihood that the current input vector would belong to the class it is classifying for. The definition of this relative likelihood value differs from one classifier to the next and is difficult to compare directly.

However, there exists generalizations of Fisher's LDA to multiple classes, which may better handle this issue. These generalizations involve altering the calculations for both within-class and between-class scatter functions, better accommodating the properties that naturally come with multiple class classification problems.

Finally, the K-nearest neighbor classifier performs relatively quite well, as it generally will when enough data is available to generalize with.

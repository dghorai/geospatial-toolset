"""
Data preparation for machine learning is the process of cleaning, transforming, and organizing 
raw data into a format that machine learning algorithms can understand.

Step-by-step guide to data preparation:
1) Data Collection
    Start with collecting data from various sources like databases, spreadsheets, or APIs.

2) Data Cleaning
    Next, clean the data by removing or correcting missing values, outliers, or inconsistencies.
    - Handling missing values:
        - Imputation method (remove records if its introduce bias, mean or median for random, forward-fill or backward-fill for time-series)
    - Handling outliers:
        - apply z-score normalization to identify this
    - Handling inconsistencies:
        - identify naming inconsistency in the database and fix
    - duplicate

3) Data Exploration
    Explore the data to gain insights into its distribution, relationships between features, and any outliers. Use visualization tools to help identify patterns, anomalies and trends.
    - EDA Analysis

4) Data Annotation
    This step is also optional, but it's important when working with image, video or audio data. Annotating the data is the process of labeling the data, for example, by bounding boxes, polygons, or points, to indicate the location of objects in the data.
    - VGG Image Annotator
    - QGIS, etc.

5) Data preprocessing / Transformation
    Then, transform the data through processes like normalization and encoding to make it compatible with machine learning algorithms.
    - Feature scaling:
        - min-max scaling
        - standardization
    - Feature encoding:
        - one-hot encoding
        - label encoding

6) Data Reduction
    Finally, reduce the data's complexity without losing the information it can provide to the machine learning model, often using techniques like dimensionality reduction.
    - dimensionality reduction

7) Data Spliting
    The last step in preparing your data for machine learning is splitting it into different sets: training, validation, and test sets.
    A common practice is using a 70-30 or 80-20 ratio for training and test sets. 

8) Data Augmentation
    This step is optional, but it can help to improve the model's performance by creating new examples from the existing data. This can include techniques such as rotating, flipping, or cropping images.
    - Apply this on training data
    

Reference:
https://www.pecan.ai/blog/data-preparation-for-machine-learning/
https://www.geeksforgeeks.org/how-to-prepare-data-before-deploying-a-machine-learning-model/
https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html

"""


def download():
    pass

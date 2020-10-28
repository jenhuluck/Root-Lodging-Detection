## Introduction
Root lodging in corn is defined as the displacement of roots from their natural vertical positions; which may lead to poor plant growth and development. Root lodging hampers the overall quality of the corn, reduces yield, and presents difficulties during harvest. In order to combat this issue, farmers are required to manually inspect each corn plot; which is a time consuming and expensive process. Corn farmers also measure the percentage of lodging and label each plot with their respective lodged percentage. Due to the natural vertical growth of corn plants after lodging, accurately detecting corn lodging has proven to be a difficult and tedious task.

## Data Source
The image data came from photos taken by an agriculture company. The images were first processed to gray scale pictures. Total number of image is 113, with 44 of them are corns with two rows and rest are corns with four rows. The raw images were protected, so only processed images were presented.

## Apporach
  * Convert RGB images to gray scale images.
  * Divide images to different number of segments and obatin density array for each image.
  * Data augmentation by concatenant 2 row image data and reset label.
  * Prepare the train and test dataset by different number of segments.
  * Apply Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting models on the new dataset.
  
## Workflow
![alt text](https://github.com/jenhuluck/Root-Lodging-Detection/blob/master/root%20lodging.png?raw=true)


## Dependencies
* Python 3.7
* sklearn 0.21.2
* pandas 1.0.5
* numpy 1.16.5

## Comparasions by models and number of segments (By accuracies)
| seg_num | Logistic Regression | Naive Bayes | KNN    | Decision Tree | Random Forest | SVM    | Gradient Boosting |
|---------|---------------------|-------------|--------|---------------|---------------|--------|-------------------|
| 10      | 58.96%              | 64.62%      | 78.77% | 94.81%        | 97.17%        | NA     | 96.23%            |
| 20      | 63.21%              | 66.50%      | 80.66% | 94.81%        | 96.77%        | 96.22% | 96.70%            |
| 30      | 77.36%              | 68.87%      | 80.66% | 93.40%        | 97.17%        | 96.22% | 97.17%            |
| 40      | 82.08%              | 70.75%      | 82.54% | 98.58%        | 96.70%        | 93.87% | 96.70%            |
| 50      | 87.74%              | 75.00%      | 81.13% | 95.75%        | 97.64%        | 84.43% | 96.70%            |
| 60      | 92.92%              | 73.11%      | 80.66% | 96.70%        | 97.64%        | 70.28% | 95.28%            |
| 70      | 94.34%              | 73.11%      | 82.55% | 94.34%        | 95.75%        | 57.55% | 95.28%            |
| 80      | 95.28%              | 72.16%      | 82.08% | 94.81%        | 95.75%        | 47.64% | 97.64%            |
| 90      | 94.81%              | 78.30%      | 85.84% | 93.39%        | 96.70%        | 47.64% | 96.23%            |
| 100     | 95.75%              | 78.30%      | 81.60% | 94.81%        | 96.70%        | 47.64% | 97.64%            |

## Usage
* drive.py is for converting raw images to gray scale images in batch mode.
* modeling.py is for data augmentation, data preparation and modeling.
* Raw images were not included. Gray scales images were provided. 




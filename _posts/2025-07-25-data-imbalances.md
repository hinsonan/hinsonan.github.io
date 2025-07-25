---
layout: post
title: "A Well Balanced Dataset Doesn't Exist: Handling Imbalances in Data"
date: 2025-07-25
categories: ML
---

They say a well balanced diet is healthy but all I see are fat people. This is how I feel when all your textbooks and classrooms give you this nice clean dataset to start with. This does not happen often in the real world. Let's address some ways to handle imbalances.

## Credit Card Fraud

We are going to use a credit card fraud dataset. This is a good choice since most of the purchases in the dataset are normal and only a small fraction of the purchases are frauds. You can find this dataset on [kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

Let's take a look at this dataset. It is important to note that the features in this dataset are from PCA being applied to it. What we are viewing are the principal components.

```
| Time | V1 | V2 | V3 | V4 | V5 | Amount | Class |
|------|----|----|----|----|----|----|-------|
| 0.0 | -1.360 | -0.073 | 2.536 | 1.378 | -0.338 | 149.62 | 0 |
| 0.0 | 1.192 | 0.266 | 0.166 | 0.448 | 0.060 | 2.69 | 0 |
| 1.0 | -1.358 | -1.340 | 1.773 | 0.380 | -0.503 | 378.66 | 0 |
| 1.0 | -0.966 | -0.185 | 1.793 | -0.863 | -0.010 | 123.50 | 0 |
| 2.0 | -1.158 | 0.878 | 1.549 | 0.403 | -0.407 | 69.99 | 0 |

*Note: Full dataset contains V1-V28 PCA components.*
```

### Data Imbalance

Let's see the ratio of normal vs fraud charges

```python
# Analyze class distribution
print("=== CLASS DISTRIBUTION ANALYSIS ===")

# Count values
class_counts = df['Class'].value_counts().sort_index()
print("Class counts:")
print(f"Normal transactions (Class 0): {class_counts[0]:,}")
print(f"Fraudulent transactions (Class 1): {class_counts[1]:,}")

# Calculate percentages and imbalance ratio
normal_pct = (class_counts[0] / len(df)) * 100
fraud_pct = (class_counts[1] / len(df)) * 100
imbalance_ratio = class_counts[0] / class_counts[1]

print(f"\nClass distribution:")
print(f"Normal: {normal_pct:.3f}%")
print(f"Fraud: {fraud_pct:.3f}%")
print(f"\nImbalance ratio: {imbalance_ratio:.0f}:1")
print("This means for every 1 fraudulent transaction, there are {:.0f} normal transactions".format(imbalance_ratio))
```

```
=== CLASS DISTRIBUTION ANALYSIS ===
Class counts:
Normal transactions (Class 0): 284,315
Fraudulent transactions (Class 1): 492

Class distribution:
Normal: 99.827%
Fraud: 0.173%

Imbalance ratio: 578:1
This means for every 1 fraudulent transaction, there are 578 normal transactions
```

Now lets calculate some [correlation statistics](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient). This is important to know which PCA components contribute the most to fraud charges

<details markdown="1">
<summary>Pearson Correlation Explained</summary>

**Understanding Correlation**

When we compute correlations in our fraud detection analysis, we're using the **Pearson correlation coefficient**. This mathematical measure helps us quantify how strongly each feature relates to fraud occurrence.

**The Mathematical Definition**

For two variables X and Y, the Pearson correlation coefficient is defined as:

$$r_{X,Y} = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}$$

Where:
- $\text{cov}(X,Y)$ is the covariance between X and Y
- $\sigma_X$ and $\sigma_Y$ are the standard deviations of X and Y  
- $\mu_X$ and $\mu_Y$ are the means of X and Y
- $E[\cdot]$ denotes the expected value

**Sample Correlation Formula**

For our dataset with `N` transactions, this becomes:

$$r_{X,Y} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Where $\bar{x}$ and $\bar{y}$ represent the sample means.

**Applied to Fraud Detection**

When we run `df.corr()['Class'].abs()`, we're calculating:

$$r_{V_j,\text{Class}} = \frac{\sum_{i=1}^{n}(v_{j,i} - \bar{v_j})(c_i - \bar{c})}{\sqrt{\sum_{i=1}^{n}(v_{j,i} - \bar{v_j})^2}\sqrt{\sum_{i=1}^{n}(c_i - \bar{c})^2}}$$

For each PCA component $V_j$ against our fraud class labels.

**Key Properties**

The correlation coefficient has these important characteristics:

- **Range**: $r \in [-1, 1]$
- **Perfect positive correlation**: $r = 1$ 
- **Perfect negative correlation**: $r = -1$
- **No linear relationship**: $r = 0$

</details>

Here are plots showing top 10 correlations and correlations of every feature.

![top10](/assets/images/top_10_correlations.png)

![correlations](/assets/images/correlations.png)

Based on these plots features `V14` and `V17` have the highest correlation with the fraud class

### Strap On Your Sampling Britches

There are a lot of techniques one can use to help alleviate this imbalance. Remember we have a highly unbalanced dataset

```
Class counts:
Normal transactions (Class 0): 284,315
Fraudulent transactions (Class 1): 492
```

The two easiest ways are random oversampling and random undersampling.

#### Random Sampling

**Over Sampling**: This method duplicates the existing minority class until classes are balanced.

**Under Sampling**: Removes majority class until classes are balanced.

Both of these methods can be tuned to where you do not have to have a perfect balance. Let's apply these to our dataset and see what happens

![over_under_sampling](/assets/images/over_under_sampling.png)

![over_under_comparison](/assets/images/over_under_comparisons.png)


```markdown
**Mean Values Comparison:**
| Feature   |   Original Normal |   Original Fraud |   Oversampled Normal |   Oversampled Fraud |   Undersampled Normal |   Undersampled Fraud |
|:----------|------------------:|-----------------:|---------------------:|--------------------:|----------------------:|---------------------:|
| Time      |         94838.2   |        80746.8   |            94838.2   |           80663.8   |             95052.8   |            80746.8   |
| Amount    |            88.291 |          122.211 |               88.291 |             121.534 |                80.348 |              122.211 |
| V14       |             0.012 |           -6.972 |                0.012 |              -6.971 |                 0.017 |               -6.972 |
| V17       |             0.012 |           -6.666 |                0.012 |              -6.664 |                 0.008 |               -6.666 |


**Standard Deviation Comparison:**
| Feature   |   Original Normal |   Original Fraud |   Oversampled Normal |   Oversampled Fraud |   Undersampled Normal |   Undersampled Fraud |
|:----------|------------------:|-----------------:|---------------------:|--------------------:|----------------------:|---------------------:|
| Time      |         47484     |        47835.4   |            47484     |           47782.1   |             47120.6   |            47835.4   |
| Amount    |           250.105 |          256.683 |              250.105 |             254.046 |               178.03  |              256.683 |
| V14       |             0.897 |            4.279 |                0.897 |               4.268 |                 0.901 |                4.279 |
| V17       |             0.749 |            6.971 |                0.749 |               6.958 |                 0.683 |                6.971 |
```

Congrats we have solved the imbalanced dataset problem. Our dataset is now balanced but this is technique introduces other problems. There is no free lunch in ML. We are bound to the forces of this world. Now this might work for your dataset and solve your issues. Let me list a few potential issues with this approach

**Oversampling Issues**

- Overfitting risk since the model may memorize duplicated samples
- No new information was added to the training set
- Dataset has increased in size

**Undersampling Issues**

- Tons of information is lost in this case due to removing large amounts of the normal class
- Risk of underfitting
- May have removed important data points

### Bring in the Synthetic

Synthetic data can be blessing and a curse. It should be approached with caution because it can cause more problems than it can solve in certain domains. You need to understand you are introducing new data to the dataset but that new data is not real. It does not fit the exact same patterns and features as the real data. In many computer vision domains synthetic data can cause your models to perform worse in the real world. If you are working in some niche area like medical X ray or sonar you have to be extra careful making synthetic data and honestly you should try to avoid it if possible.

In other domains synthetic data can be very beneficial. This domain of anomaly detection with tabular credit charges it can be an effective tool. My one caveat is that no matter the domain you need to physically look at your synthetic data and be aware of the potential issues it could cause. If it shifts all the statistics in a major way from the real data you may want to evaluate if your new data is viable.

### Throw Caution to the Wind: Data go BRRR

**SMOTE (Synthetic Minority Oversampling Technique)**

[Paper](https://arxiv.org/pdf/1106.1813)

This method came out around 2011. This is from the paper and a good summary of how it works


>We propose an over-sampling approach in which the minority class is over-sampled by creating “synthetic” examples rather than by over-sampling with replacement. This approach
>is inspired by a technique that proved successful in handwritten character recognition (Ha
>& Bunke, 1997). They created extra training data by performing certain operations on
>real data. In their case, operations like rotation and skew were natural ways to perturb
>the training data. We generate synthetic examples in a less application-specific manner, by
>operating in “feature space” rather than “data space”. The minority class is over-sampled
>by taking each minority class sample and introducing synthetic examples along the line
>segments joining any/all of the k minority class nearest neighbors. Depending upon the
>amount of over-sampling required, neighbors from the k nearest neighbors are randomly
>chosen. Our implementation currently uses five nearest neighbors. For instance, if the
>amount of over-sampling needed is 200%, only two neighbors from the five nearest neighbors are chosen and one sample is generated in the direction of each. Synthetic samples
>are generated in the following way: Take the difference between the feature vector (sample)
>under consideration and its nearest neighbor. Multiply this difference by a random number
>between 0 and 1, and add it to the feature vector under consideration. This causes the
>selection of a random point along the line segment between two specific features. This
>approach effectively forces the decision region of the minority class to become more general.

Duplication strategies have their issues and this method can be a great tool to try. I will summarize some of the key benefits.

* Tries to make the minority class more general in its features compared to being extremely specific
* The examples that this method generate should be along line segments between the minority neighbors
* Attempts to fix the overfitting issue of duplication

In order to help visualize this I will use PCA to reduce this dataset down to two features just so it appears neatly on a plot. You could choose only 2 or 3 of the features to plot but when you run PCA with two components the plot looks nice and you can see the effects of SMOTE

![SMOTE_DATA](/assets/images/smote_comparison.png)

![SMOTE_FEATURE_COMPARISON](/assets/images/smote_features.png)

```
=== TOP 5 FEATURES WITH LARGEST DIFFERENCES ===
Largest MEAN differences:
  V14: Original=-7.3060, Synthetic=-8.0206, Diff=0.7145
  V12: Original=-7.6222, Synthetic=-8.2018, Diff=0.5796
  V2: Original=2.8113, Synthetic=3.2466, Diff=0.4354
  V10: Original=-4.8290, Synthetic=-5.2241, Diff=0.3950
  V3: Original=-4.5798, Synthetic=-4.9639, Diff=0.3841

Largest STD DEV differences:
  V2: Original=1.7124, Synthetic=0.8127, Diff=0.8996
  V17: Original=3.9913, Synthetic=3.1306, Diff=0.8606
  V12: Original=2.5273, Synthetic=1.8131, Diff=0.7142
  V11: Original=2.0488, Synthetic=1.3462, Diff=0.7026
  V3: Original=2.2002, Synthetic=1.5492, Diff=0.6510
```

This method is a great tool to try and get more reasonable minority classes. You can use this tool in your ML experiments and see if it helps with your dataset and domain. Keep in mind that this method is still not adding real world data points and relying on synthetic data can bias your model. Here is the cool thing. This is just one of the methods. There are other methods that have built upon SMOTE and may be better for your problem.

### Other Methods

List of other popular methods:

#### Over Sampling Methods

- **[ADASYN](https://www.researchgate.net/publication/224330873_ADASYN_Adaptive_Synthetic_Sampling_Approach_for_Imbalanced_Learning)** - Adaptive Synthetic Sampling generates more synthetic data for minority class samples that are deemed harder to learn, using density distribution as a way to pick the harder classes.

- **[BorderlineSMOTE](https://www.mdpi.com/1996-1073/15/13/4751)** - Variant of SMOTE That uses the borderline data points in order to create synthetic data points.

- **SVMSMOTE** - Uses an SVM algorithm (obviously) to detect samples for generating new synthetic samples.

#### Undersampling Methods

- **[NearMiss](https://www.sciencedirect.com/science/article/pii/S0957417422005280)** - [Controlled undersampling with 3 versions](https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/): "NearMiss-1: Majority class examples with minimum average distance to three closest minority class examples. NearMiss-2: Majority class examples with minimum average distance to three furthest minority class examples. NearMiss-3: Majority class examples with minimum distance to each minority class example."


- **[TomekLinks](https://www.researchgate.net/publication/326590590_Classification_of_Imbalance_Data_using_Tomek_Link_T-Link_Combined_with_Random_Under-sampling_RUS_as_a_Data_Reduction_Method)** - Removes Tomek's links - pairs of samples from different classes that are each other's nearest neighbors

- **[EditedNearestNeighbours](https://www.researchgate.net/publication/220870570_Edited_Nearest_Neighbor_Rule_for_Improving_Neural_Networks_Classifications)** - Removes samples if most of their k-nearest neighbors belong to a different class.

## Combination Methods

- **[SMOTEENN](https://www.researchgate.net/publication/346282224_SMOTE-ENN-Based_Data_Sampling_and_Improved_Dynamic_Ensemble_Selection_for_Imbalanced_Medical_Data_Classification)** - Combines SMOTE oversampling with Edited Nearest Neighbours cleaning to both generate synthetic samples and remove similiar ones that could add more noise.
- **[SMOTETomek](https://arxiv.org/html/2501.06491v1)** - Combines SMOTE oversampling with Tomek links cleaning to generate synthetic samples and have more distinct samples.

### Example Code Using These Methods

I highly recommend `imblearn` library in python for experimenting and playing with these methods.

Here is some code you can build off of to try these methods out

```python
# Imbalanced-Learn Methods Demo
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.1, 0.9], 
                          n_informative=3, n_redundant=1, random_state=42)
print(f"Original: {Counter(y)}")

# Oversampling Methods
methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'SVMSMOTE': SVMSMOTE(random_state=42),
    'RandomOverSampler': RandomOverSampler(random_state=42),
    
    # Undersampling Methods
    'RandomUnderSampler': RandomUnderSampler(random_state=42),
    'NearMiss': NearMiss(version=1, n_neighbors=3),
    'TomekLinks': TomekLinks(),
    'EditedNearestNeighbours': EditedNearestNeighbours(),
    
    # Combination Methods
    'SMOTEENN': SMOTEENN(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42)
}

# Apply each method and show results
for name, sampler in methods.items():
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f"{name:20}: {Counter(y_resampled)}")
```

## With Great Power...

You can use all these tools to help solve your data problems but no tool is a silver bullet. Some of these methods may make your model perform worse. It is up to you to experiment and understand these methods when applied to your domain. These methods are more tailored toward tabular data or vector data. If you have image data then you have to use some other techniques and honestly the problem exponentially increases in difficulty. We may explore image data imbalances later. Keep in mind that these methods are all working on the data. We have not talked about model specific algorithms that can help with imbalances. This is something else we can explore later. If you are interested in a certain topic make sure to contact me and let me know what you want covered next.

[full code for the visuals](https://github.com/hinsonan/hinsonan.github.io/blob/master/code_examples/imbalanced_dataset/imbalanced_datasets.ipynb)
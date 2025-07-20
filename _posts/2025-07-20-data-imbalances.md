---
layout: post
title: "A Well Balanced Dataset Doesn't Exist: Handling Imbalances in Data"
date: 2025-07-20
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

Here are plots showing top 10 correlations and correlations of every feature

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
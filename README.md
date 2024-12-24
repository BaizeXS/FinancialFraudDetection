# Online Transaction Fraud Detection Report

## 1. Overview

In the rapid development of e-commerce today, the detection and prevention of fraudulent transactions have become an important and complex challenge. Fraudulent activities not only cause financial losses to consumers but also have a significant impact on merchants' income and reputation. At the same time, existing fraud detection systems, while reducing fraud, may also lead to the rejection of normal transactions due to false positives, thereby affecting customer experience. How to improve the accuracy of fraud detection while reducing false positives has become an urgent problem to be solved in the industry. The core goal of this task is to predict whether an online transaction is fraudulent by constructing and optimizing machine learning models.

## 2. Exploratory Data Analysis and Data Preprocessing

### 2.1 Dataset Basic Information

The dataset used in this study consists of 100,000 transaction records and 101 features, with the aim of identifying fraudulent activities in online transactions through analysis and modeling. The target variable `isFraud` reveals that non-fraudulent transactions account for 88.68% and fraudulent transactions for 11.32%, highlighting a class imbalance issue. Features are categorized into categorical, integer, logical, and numerical types. Key features include `TransactionID`, `TransactionDT`, and `TransactionAmt`. The dataset is complex, containing many anonymized features and categorical variables, which demand higher standards for data preprocessing and modeling.

### 2.2 Missing Value Analysis and Handling

The dataset contains a total of 3,481,741 missing values across 78 features. Multiple strategies were employed to handle the missing data: features with a missing rate exceeding 90%, such as `dist1` and `D11`, were directly removed; other features were imputed based on their types and business logic. Categorical variables were filled with "unknown", numerical variables with the median, and logical variables with the mode. All missing values were successfully imputed, ensuring data integrity and laying the groundwork for subsequent modeling.

### 2.3 Univariate Analysis

During the univariate analysis phase, we conducted basic statistical and visual analyses of the distributions of numerical and categorical variables in the dataset. The analysis of numerical variables revealed that most transaction amounts were small, while large transactions were rare, suggesting the need to pay special attention to potential anomalies in large transactions during subsequent analyses. For categorical variables, bar charts highlighted unusually high transaction volumes for certain product types and email domains, which might be associated with fraudulent behavior. Additionally, we generated frequency tables and missing value charts to facilitate further data analysis and cleaning, laying a foundation for multivariate analysis and model building.

### 2.4 Bivariate Analysis

In the bivariate analysis, we further explored the relationships between various variables and fraudulent behavior. The analysis of numerical variables revealed that large transactions were more likely to be fraudulent, while transaction time had little correlation with fraudulent behavior. For categorical variables, we found that certain card types and device types were strongly associated with fraudulent behavior, providing important clues for identifying fraud. These analysis results are critical for subsequent feature selection and model optimization, helping to improve the accuracy and efficiency of fraud detection.

### 2.5 Correlation Analysis

In correlation analysis, we focused on numerical variables and identified variable pairs with correlation coefficients greater than 0.7 by calculating the correlation matrix. This helped us detect potential redundant features and dependencies between variables. The target variable `isFraud` and the unique identifier `TransactionID` were excluded, and correlation computations were performed on the remaining numerical variables. The analysis revealed some highly correlated variable pairs, such as `TransactionDT` and its derived variable `TransactionDT..Hour.`, as well as `C4` and `C6`, which were nearly identical. Additionally, we identified strongly correlated variables within the `C` and `D` series, as well as cross-category correlations, such as those between `card3` and `id_17`. These findings are crucial for reducing multicollinearity in the model and optimizing feature engineering.

### 2.6 Outlier Handling

For outlier handling, we used the IQR method to identify and address outliers in the data. A threshold of 1.5 times the IQR was applied, and the columns `TransactionID` and `isFraud`, which were unsuitable for outlier analysis, were excluded. The data was split into fraudulent and non-fraudulent samples, and outlier handling was performed exclusively on the non-fraudulent samples. By calculating the quartiles of each numerical variable, values beyond the boundaries were identified as outliers and replaced. For example, variables such as `TransactionAmt` and `addr1` were found to have a large number of outliers, all of which were replaced. This process significantly reduced the potential impact of extreme values on the model, ensured data stability, and provided a reliable foundation for subsequent modeling.

### 2.7 Feature Engineering

In this study on online transaction fraud detection, we implemented a series of feature engineering strategies to enhance the model's predictive performance. First, we extracted multiple time-related features from the transaction time data, such as hour, day of the week, and day of the month, which helped uncover periodic patterns in transaction behavior. Second, we applied logarithmic transformation and binning to transaction amounts to reduce data skewness and make the distribution closer to normal, enabling the model to better understand the distribution characteristics across different transaction amount ranges. We also normalized authentication fields, mapping their values to a range of 0 to 1, standardizing feature scales and simplifying the model training process. Additionally, we converted logical fields into numerical values to ensure all features were presented in a numerical format, allowing the model to process them directly. These feature engineering efforts not only enriched the dataset's feature set but also enhanced the model's ability to identify transaction behavior patterns, laying a solid foundation for model training and optimization.

### 2.8 Categorical Feature Encoding

When handling categorical variables in online transaction fraud detection, we employed multiple encoding strategies, including target encoding, one-hot encoding, and binary encoding, to retain the original information of the data while improving the model's learning efficiency. A total of 18 categorical variables were identified and categorized based on the number of categories, business relevance, and data distribution. For high-cardinality variables, we used smoothed target encoding to reduce the risk of overfitting. For business-relevant variables and those with a moderate number of categories, we applied target encoding. Binary variables were converted into 0 and 1 using binary encoding, while low-cardinality variables and binned transaction amount variables were encoded using one-hot encoding. This encoding process successfully transformed all categorical variables into numerical formats, providing high-quality input data for model training and further improving the model's predictive accuracy and generalization ability.

### 2.9 Feature Selection

In this study on online transaction fraud detection, we conducted meticulous feature selection to reduce data complexity by removing redundant or low-value features, thereby improving model performance and training speed. First, we identified a set of protected columns, including the target variable `isFraud`, one-hot encoded features, time-related features, and some business-critical encoded variables, which were not removed during the feature selection process. Next, we detected and removed 37 zero-variance features, which had constant or nearly constant values throughout the dataset. We also performed correlation analysis on numerical features and removed redundant features with a correlation exceeding the 0.8 threshold, such as `ProductCD`, `id_30`, `id_34`, and `id_28`. Ultimately, we retained 51 features out of the original 100. This feature selection significantly reduced data redundancy while preserving key features and business-relevant information, providing a more efficient and reliable input dataset for subsequent model training.

### 2.10 Feature Scaling

In the online transaction fraud detection study, we applied feature scaling to numerical features to ensure compatibility with model input requirements. Using the robust scaling method, which relies on the median and interquartile range (IQR), we minimized the impact of outliers. Features like the target variable, one-hot encoded features, time-related features, and already encoded categorical/binary variables were excluded from scaling. A total of 19 numerical features, including transaction time, transaction amount, and user/card/address information, were scaled using the formula:

$$
x' = \frac{x - median(x)}{IQR(x)} .
$$

Compared to standardization and min-max scaling, robust scaling is less sensitive to outliers. This process improved data consistency, enhanced the model's adaptability to features of varying dimensions, and reduced the influence of outliers. The scaled dataset provided a solid foundation for model training, improving both performance and generalization.

### 2.11 Imbalanced Data Handling

In this online transaction fraud detection project, we addressed the issue of class imbalance, where fraud samples were significantly fewer than non-fraud samples. Without correction, models often become biased toward the majority class, reducing their ability to detect fraud. To resolve this, we applied three data balancing methods: SMOTE, ROSE, and ADASYN.

The original dataset had 70,922 non-fraud samples (88.7%) and only 9,078 fraud samples (11.3%), resulting in a 7.81:1 imbalance ratio. We implemented the following methods:

1. **SMOTE**: Synthetic Minority Oversampling Technique generated additional fraud samples, increasing their count to 36,312 (33.9% of total), reducing the imbalance ratio to 1.95:1.
2. **ROSE**: This method created synthetic samples near the minority class and undersampled the majority class, balancing the dataset with fraud and non-fraud samples at 40,057 and 39,943, respectively (1:1 ratio).
3. **ADASYN**: By focusing on low-density regions of fraud samples, this method increased their count to 71,122, matching non-fraud samples at 70,922, achieving a near 1:1 ratio.

These methods effectively alleviated class imbalance. ROSE and ADASYN achieved complete balance, while SMOTE moderately reduced the bias. The best method can be chosen based on business needs and model testing.

In conclusion, addressing class imbalance significantly improved the model’s ability to detect fraud and reduced bias toward the majority class. The balanced dataset provided a solid foundation for improving fraud detection accuracy and recall.

## 3. Model Construction and Training

In this online transaction fraud detection project, we tested various models to address this complex classification problem, including traditional statistical methods, modern ensemble algorithms, and deep learning techniques to cater to different data characteristics and business requirements. Due to the low training efficiency of Support Vector Machines (SVM) when handling high-dimensional data and large-scale samples, we decided not to use SVM. Below is an overview of the models we employed:

### 3.1 Logistic Regression

Logistic Regression is a simple and effective linear model suitable for handling linearly separable binary classification problems. We introduced Lasso and Ridge regularization during training to mitigate multicollinearity issues and prevent overfitting. Five-fold cross-validation was employed to optimize performance, and parallel processing was utilized to speed up training. Logistic Regression became an important baseline model in our project due to its fast training speed and good interpretability.

### 3.2 Random Forest

Random Forest is a popular ensemble learning method that constructs multiple decision trees and combines them through a voting mechanism for classification. We configured the Random Forest model with 100 trees and enabled feature importance analysis. To improve computational efficiency, we utilized a parallel Random Forest implementation, fully leveraging a multi-core computing environment. Random Forest performed exceptionally well in handling non-linear and high-dimensional features, making it one of the core models for our fraud detection task.

### 3.3 XGBoost

XGBoost is a highly efficient gradient boosting algorithm with strong feature processing capabilities and excellent model generalization. During training, we set 200 iterations, a learning rate of 0.05, and a maximum depth of 8, while introducing row sampling and column sampling to enhance model robustness. Multi-threaded support significantly sped up the training process. XGBoost was one of the main models in our project and showed outstanding performance, particularly in detecting fraudulent samples.

### 3.4 LightGBM

LightGBM is an efficient gradient boosting framework, particularly suitable for handling large-scale datasets and high-dimensional features. We configured it with 64 leaf nodes, a learning rate of 0.05, and 200 training iterations. By leveraging GPU or multi-core parallel training, we significantly improved training efficiency, making it one of the preferred methods for processing large fraud detection datasets.

### 3.5 Neural Network

To enhance the model's ability to capture nonlinear features, we built a simple feedforward neural network with a single hidden layer containing 10 nodes and applied L2 regularization to prevent overfitting. This model is suitable for tasks with fewer features but is limited compared to more complex deep learning models, making it more appropriate as an alternative solution.

### 3.6 Deep Neural Network (DNN)

We designed a deep neural network with three hidden layers containing 64, 32, and 16 nodes, respectively. The model utilized the Sigmoid activation function and the `rprop+` optimization method, with a maximum of 1 million iterations. DNN demonstrated excellent performance in modeling complex feature interactions, significantly improving the classification accuracy for fraud detection and offering strong advantages in handling high-dimensional nonlinear features.

### Unused Model: SVM

Although SVM theoretically has strong classification capabilities for small-scale data and nonlinear problems, its training time was excessively long due to the large dataset size and high feature dimensionality in this project. We tested the linear kernel of SVM with parallel support enabled, but the training speed still failed to meet practical requirements. As a result, SVM was not selected as a practical model for this project.

In this project, we evaluated various models, including Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Networks, and Deep Neural Networks. Each model has its strengths: Logistic Regression is fast and easy to interpret; Random Forest and gradient boosting trees (XGBoost and LightGBM) are stable and have strong feature processing capabilities; Neural Networks (especially DNN) excel at capturing complex nonlinear relationships between features. Based on business needs and model performance, we selected models better suited for handling large-scale data and high-dimensional features for deployment. While SVM was excluded due to its long training time, the other models provided a solid foundation for the final detection solution.

## 4 Model Evaluation

In our project, we evaluated model performance using the `evaluate_models` function on a 20,000-sample test set with metrics like Accuracy, Precision, Recall, F1 Score, and AUC, and used confusion matrices for clearer insights. The `predict_model` function generated predictions, with Logistic Regression and Random Forest basing their predictions on probability values and Neural Networks providing direct probability outputs. We then calculated these metrics and analyzed performance in detail with the `calculate_metrics` function. Finally, the `print_results` function helped us compare model performances easily, allowing us to choose the best model for our needs.

### 4.1 Model Evaluation Metrics Overview

| Model Name              | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) | Training Time (mins) | Data Balancing Method | Remarks                                   |
| ----------------------- | ------------ | ------------- | ---------- | ------------ | ------- | -------------------- | --------------------- | ----------------------------------------- |
| **Logistic Regression** | 93.10        | 68.14         | 72.10      | 70.07        | 91.62   | 1.20                 | SMOTE                 | Balanced performance, baseline model.     |
| **Logistic Regression** | 81.47        | 35.35         | 78.97      | 48.84        | 88.45   | 0.28                 | ROSE                  | High recall, low precision.               |
| **Logistic Regression** | 80.07        | 33.76         | 80.94      | 47.64        | 87.10   | 0.98                 | ADASYN                | Lowest precision, moderate recall.        |
| **Random Forest**       | 99.42        | 99.91         | 94.91      | 97.34        | 97.45   | 0.04                 | None (Original Data)  | Exceptionally high accuracy/precision.    |
| **XGBoost**             | 99.41        | 100.00        | 94.73      | 97.29        | 99.79   | 0.07                 | None (Original Data)  | Near-perfect precision, very high AUC.    |
| **LightGBM**            | 99.48        | 99.86         | 95.45      | 97.60        | 99.84   | 0.05                 | None (Original Data)  | Best overall performance, recommended.    |
| **Deep Neural Network** | 97.98        | 89.47         | 92.90      | 91.15        | 98.07   | 5.56                 | SMOTE                 | Strong performance, but training is slow. |
| **Deep Neural Network** | 96.16        | 100.00        | 65.71      | 79.31        | 91.67   | 0.19                 | ROSE                  | High precision, but recall dropped.       |
| **Deep Neural Network** | 96.86        | 81.22         | 93.62      | 86.98        | 98.52   | 46.82                | ADASYN                | High AUC, but inefficient training time.  |
| **Neural Network**      | 11.20        | 11.20         | 100.00     | 20.14        | 56.25   | 0.19                 | SMOTE                 | Overfitting or severe setup issues.       |
| **Neural Network**      | 11.20        | 11.20         | 100.00     | 20.14        | 53.73   | 0.19                 | ROSE                  | Same performance as SMOTE, problematic.   |
| **Neural Network**      | 11.20        | 11.20         | 100.00     | 20.14        | 53.77   | 0.15                 | ADASYN                | Same issues across all balancing methods. |

### 4.2 Confusion Matrix Analysis

1. **Logistic Regression with SMOTE**

   |              |        Predicted 0         |        Predicted 1        |
   | :----------: | :------------------------: | :-----------------------: |
   | **Actual 0** | True Negative (TN): 17,005 | False Positive (FP): 755  |
   | **Actual 1** |  False Negative (FN): 625  | True Positive (TP): 1,615 |

   The model generated 755 false positives and 625 false negatives. The results indicate that the model has a relatively high false positive rate, while the recall is at a relatively moderate level, reaching 72.10%.

2. **Random Forest**

   |              |        Predicted 0         |        Predicted 1        |
   | :----------: | :------------------------: | :-----------------------: |
   | **Actual 0** | True Negative (TN): 17,758 |  False Positive (FP): 2   |
   | **Actual 1** |  False Negative (FN): 114  | True Positive (TP): 2,126 |

   The number of false positives was only 2, while there were 114 false negatives, indicating that the model has a very low false positive rate. The recall reached as high as 94.91%, demonstrating that the model's classification performance is quite reliable.

3. **XGBoost**

   |              |        Predicted 0         |        Predicted 1        |
   | :----------: | :------------------------: | :-----------------------: |
   | **Actual 0** | True Negative (TN): 17,760 |  False Positive (FP): 0   |
   | **Actual 1** |  False Negative (FN): 118  | True Positive (TP): 2,122 |

   The model achieved zero false positives, demonstrating excellent performance in scenarios requiring minimal tolerance for false positives. However, it produced 118 false negatives, resulting in a recall of 94.73%, which is slightly below ideal. Despite this, the model is highly effective for low false-positive tolerance applications.

4. **LightGBM**

   |              |        Predicted 0         |        Predicted 1        |
   | :----------: | :------------------------: | :-----------------------: |
   | **Actual 0** | True Negative (TN): 17,757 |  False Positive (FP): 3   |
   | **Actual 1** |  False Negative (FN): 102  | True Positive (TP): 2,138 |

   The model generated 3 false positives and 102 false negatives, demonstrating a good balance between controlling false positives and reducing false negatives. Its recall reached 95.45%, making it one of the best-performing models.

5. **Deep Neural Network (DNN with SMOTE)**

   |              |        Predicted 0         |        Predicted 1        |
   | :----------: | :------------------------: | :-----------------------: |
   | **Actual 0** | True Negative (TN): 17,515 | False Positive (FP): 245  |
   | **Actual 1** |  False Negative (FN): 159  | True Positive (TP): 2,081 |

   The model generated 245 false positives and 159 false negatives, with a recall of 92.90%. However, the relatively high number of false positives resulted in overall performance that was inferior to the LightGBM and XGBoost models.

6. **Neural Network**

   |              |      Predicted 0       |         Predicted 1         |
   | :----------: | :--------------------: | :-------------------------: |
   | **Actual 0** | True Negative (TN): 0  | False Positive (FP): 17,760 |
   | **Actual 1** | False Negative (FN): 0 |  True Positive (TP): 2,240  |

   The model produced 17,760 false positives and no false negatives, indicating severe overfitting. It was overly optimized on the training data, losing its ability to generalize, making it unsuitable for practical use.

### 5.3 Model strengths and weaknesses analysis

After evaluating various models for online transaction fraud detection, **LightGBM** emerged as the best performer, achieving a recall of 95.45%, only 3 false positives, 102 false negatives, and an AUC of 99.84%. Its efficiency, speed, and balanced performance make it ideal for large-scale real-world applications.

As a strong alternative, **XGBoost** achieved zero false positives, 100% precision, a recall of 94.73%, and an AUC of 99.79%, making it highly suitable for scenarios requiring zero tolerance for false alarms, such as financial fraud detection or medical diagnoses.

**Random Forest** also performed well, with a recall of 94.91%, 2 false positives, and an AUC of 97.45%. Its simplicity, speed, and stability make it a reliable choice for tasks requiring both efficiency and accuracy.

The **Deep Neural Network (DNN with SMOTE)** showed promise in capturing complex patterns, achieving a recall of 92.90% and an AUC of 98.07%. However, it had a higher false positive rate (245) and longer training time, requiring further optimization for practical use.

**Logistic Regression with SMOTE** served as a simple baseline model with a recall of 72.10%, but its high false positive rate (755) and inability to model complex patterns limit its use to exploratory or lightweight tasks.

The standard **Neural Network** performed poorly, with excessive false positives (17,760) and severe overfitting, making it unsuitable for deployment.

In summary, **LightGBM** is the top recommendation, while **XGBoost** is ideal for zero false-positive tolerance scenarios. **Random Forest** is a stable secondary option, and **DNN with SMOTE** has potential after further optimization. Logistic Regression and standard Neural Networks are not recommended for complex fraud detection tasks.

## 6. Optimization

In our online transaction fraud detection project, we introduced two key systems: logging and parallel computing.

**Logging System**: Using the `logger` package, we recorded program details, warnings, and errors, saving all logs in `./logs/fraud_detection.log`. Logs were color-coded by levels (ERROR, WARN, INFO, DEBUG) for easy analysis. The log level was set to TRACE to capture detailed information for debugging and monitoring.

**Parallel Computing**: To enhance data processing and model training efficiency, we employed `doParallel` and `foreach` packages for parallel computing. The cluster was configured automatically based on CPU cores, enabling multithreaded processing. At project completion, the cluster was stopped to release resources. This strategy significantly boosted computation speed and overall efficiency.
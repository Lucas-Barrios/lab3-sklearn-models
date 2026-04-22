# Lab 3 — sklearn Model Training & Evaluation Report
**Author:** Lucas Barrios  
**Date:** April 2026

---

## Part 1: Breast Cancer Prediction

### My Approach
For this part I used the breast cancer dataset that comes built into sklearn.
It has 569 patient samples and 30 features measuring cell nucleus characteristics.
I split the data 80/20 into training and test sets, making sure both sets had
a similar proportion of malignant and benign cases using stratified sampling.
Then I trained a KNN model and tested different values of K to find the best one.

### Results
| Metric | Score |
|--------|-------|
| Training Accuracy | 94.73% |
| Test Accuracy | 91.23% |
| Precision | 94.29% |
| Recall | 91.67% |

### K Value Comparison
| K | Accuracy | Precision | Recall |
|---|----------|-----------|--------|
| 1 | 92.11% | 95.65% | 91.67% |
| 3 | 92.98% | 94.44% | 94.44% |
| 5 | 91.23% | 94.29% | 91.67% |
| 7 | 92.98% | 94.44% | 94.44% |
| 9 | 93.86% | 94.52% | 95.83% |
| 11 | 93.86% | 94.52% | 95.83% |

I chose **K=9** as the best value because it had both the highest accuracy and
the highest recall. In a medical context I think recall matters more than precision,
it's better to flag a potential cancer and be wrong than to miss it entirely.

### What I learned
Looking at the feature distributions, malignant tumors consistently had larger
radius, perimeter and area than benign ones. This made sense visually and helped
me understand why the model performed well. The features are genuinely different
between the two groups. The model missed 6 malignant cases at K=5, which reminded
me that even a 91% accurate model can still make dangerous mistakes in healthcare.

---

## Part 2: Customer Churn Prediction

### My Approach
For this part I worked independently with the Telco Customer Churn dataset.
It had 7,043 customers and 21 columns, a mix of numeric and text data.
Most of the work went into preprocessing. Converting Yes/No columns to 1/0,
fixing TotalCharges which was stored as text instead of a number, and
using get_dummies to handle columns with multiple categories like Contract type.
Once the data was clean I followed the same steps as Part 1.

### Results (K=5)
| Metric | Score |
|--------|-------|
| Training Accuracy | 82.84% |
| Test Accuracy | 76.58% |
| Precision | 57.91% |
| Recall | 43.05% |

### K Value Comparison
| K | Accuracy | Precision | Recall |
|---|----------|-----------|--------|
| 1 | 71.11% | 45.55% | 45.19% |
| 3 | 76.37% | 56.90% | 45.19% |
| 5 | 76.58% | 57.91% | 43.05% |
| 7 | 78.28% | 62.88% | 44.39% |
| 9 | 78.71% | 64.92% | 43.05% |
| 11 | 78.57% | 65.79% | 40.11% |
| 15 | 78.99% | 67.41% | 40.37% |

The best accuracy was at **K=15** with 78.99%, but I noticed that as K increased,
recall kept dropping. This was an interesting tradeoff I didn't expect.

### What I found
The churn model was noticeably harder than the cancer model. Only 26% of customers
actually churned, which means the dataset is imbalanced. The model gets rewarded
for just predicting "no churn" most of the time. The recall of 40% means the model
is missing more than half of the customers who are about to leave, which in a real
business scenario would be a serious problem.

I also noticed that preprocessing took much longer than the actual modeling.
Most of the columns were text and needed to be converted before the model
could use them. This made me understand why data cleaning is considered the
hardest part of machine learning in practice.

### What I would recommend to the company
- The model in its current state is not reliable enough to act on alone.
- A more powerful model like Random Forest would likely perform better here.
- Feature scaling should be applied since KNN is sensitive to columns with
  very different ranges of values.
- The company should focus on identifying month-to-month contract customers
  as they showed the highest churn rates in the data.

### Limitations
- I did not apply feature scaling which would have improved KNN performance.
- The class imbalance was not addressed. Techniques like SMOTE could help.
- With more time I would have tried other algorithms to compare results.
- 40% recall means the model is still missing most churners in practice.
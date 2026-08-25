# E-commerce Customer Segmentation and Prediction

Final capstone artifacts built from the supplied data.csv.

- Cleaned transactions: 392,692
- Customers after cleaning: 4,338
- Development customers: 3470 (80%)
- Final test customers: 868 (20%)
- RFM features: Recency, Frequency, Monetary
- Clustering: K-Means, Hierarchical, DBSCAN comparison
- Final deployable segmentation: K-Means, K=4 (selected for business interpretability and predictability; K=2 had the highest raw silhouette but only produced two broad groups)
- Prediction model: Logistic Regression, C=30
- Validation accuracy: 0.9942
- Validation weighted F1: 0.9942
- Final 20% test K-Means silhouette: 0.3385

## Important
Clustering is unsupervised, so the 20% set has no independent human-labelled ground truth. The final test therefore reports held-out clustering quality and classifier agreement with the frozen K-Means segment definition rather than claiming conventional classification accuracy as a true ground-truth metric.

Open `notebooks/Ecommerce_Customer_Segmentation.ipynb` for the full workflow.

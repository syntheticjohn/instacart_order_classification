# Predicting Product Re-Orders On Instacart

Predict which products Instacart users will reorder using classification algorithms (logistic regression, random forest and gradient boosting)

**Project overview:**
- Predicted which products each Instacart user will buy in the next order given a history of orders using classification algorithms and engineering time-based features to optimize F1 score 
- The ~8 million product orders from Kaggle were stored in postgreSQL 
- Trained, evaluated and compared logistic regression, random forest and gradient boosted trees models
- XGBoost was selected as the final model and trained on the full training data on AWS, and performed an F1 of 0.440 on the test set
- Results visualized with Tableau

**This repo includes:** 

- **instacart_proprocessing.py**: data preprocessing
- **instacart_modeling_subsample.ipynb**: feature engineering and modeling on subsample of data 
- **instacart_modeling.ipynb**: feature engineering and final modeling on entire training data
- **instacart_order_predictions_slides.pdf**: pdf of project presentation slides

**Note 1:** The proprocessing python script is designed to run where postgresql has already been setup and pre-stored with the instacart data (sourced from: kaggle.com/c/instacart-market-basket-analysis/data)  
**Note 2:** Some data files were excluded from the data folder due to github's size limitation

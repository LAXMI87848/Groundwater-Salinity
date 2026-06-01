% \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

% \*\*\* # Groundwater Salinity Prediction using Machine Learning

**Corresponding Author:** Dr. Laxmi Pandey (laxmigeophybhu@gmail.com)
% \*\*\* Department of Artificial Intelligence
% \*\*\* IIT Kharagpur
% \*\*\*West Bengal, India


This repository contains the complete codebase for the paper:  
*"Interpretable AI for Prediction of Groundwater Salinization in Israel"* (under review).

The pipeline includes data preprocessing, training of seven machine learning models, Double Machine Learning (DML) for conditional association analysis, and explainable AI (SHAP, GSA) for model interpretability.


% \*\*\* Source Code is mainly written for research purposes. The codes are

% \*\*\* having copyrights and required proper citations whenever it is used.
% \*\*\*Citation
% \*\*\*If you use this code, please cite the associated paper.

% \*\*\*Authors: Dr. Laxmi Pandey, Dr. Adway Mitra, and co-authors
% \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*



(Copy all set of files including data in one folder)

(After downloading the uploaded data (Dataset_to_share) and codes, give your respective path of the location\folder where the data is been saved, for the codes to run properly)
'Dataset_to_share.csv` #Processed dataset (anonymized, aggregated) 
'requirements.txt' \ #All Python dependencies with versions 
(To reproduce the environment, run: pip install -r requirements.txt)


Source Codes:

a.	ANN\_country\_model.py: This code predicts groundwater salinity (Cl⁻ concentration) at country scale using ANN model with basin density-based weighting. The model incorporates temporal validation, feature balancing, and multiple analysis techniques including XAI based SHAP and Global Sensitivity Analysis (GSA).


b.	CNN\_country\_model.py: This code predicts groundwater salinity (Cl⁻ concentration) at country scale using CNN model with basin density-based weighting. The model incorporates temporal validation, feature balancing, and multiple analysis techniques including XAI based SHAP and Global Sensitivity Analysis (GSA).


c.	LSTM\_country\_model.py: This code predicts groundwater salinity (Cl⁻ concentration) at country scale using LSTM model with basin density-based weighting. The model incorporates temporal validation, feature balancing, and multiple analysis techniques including XAI based SHAP and Global Sensitivity Analysis (GSA).


d.	XgBoost\_country\_model.py: This code predicts groundwater salinity (Cl⁻ concentration) at country scale using eXTreme Gradient Boosting model with basin density-based weighting. The model incorporates temporal validation, feature balancing, and multiple analysis techniques including XAI based SHAP and Global Sensitivity Analysis (GSA).


e.	RandomForest\_country\_model.py: This code predicts groundwater salinity (Cl⁻ concentration) at country scale using Random Forest model with basin density-based weighting. The model incorporates temporal validation, feature balancing, and multiple analysis techniques including XAI based SHAP and Global Sensitivity Analysis (GSA).


f.	LinearRegression\_country\_model.py: This code predicts groundwater salinity (Cl⁻ concentration) at country scale using Linear Regression model with basin density-based weighting. The model incorporates temporal validation, feature balancing, and multiple analysis techniques including XAI based SHAP and Global Sensitivity Analysis (GSA).


g.	TabTransformer_country_model.py: This code predicts groundwater salinity (Cl⁻ concentration) at country scale using TabTransformer model with basin density-based weighting. The model incorporates temporal validation, feature balancing, and multiple analysis techniques including XAI based SHAP and Global Sensitivity Analysis (GSA).


h.	DML.py: This code implements Double Machine Learning (DML) with cross-fitting to estimate conditional associations between environmental predictors and groundwater salinity while controlling for confounding variables. Using 5-fold cross-validation with Random Forest regressors as nuisance functions, it calculates debiased effect sizes, p-values, standard errors, and 95% confidence intervals for each predictor. The code generates comprehensive visualizations including bar plots, volcano plots, significance distributions, correlation heatmaps, and comparisons with traditional Random Forest feature importance. Results are exported to Excel files, with special focus on identifying the top 10 predictors with the lowest p-values (most statistically significant). The analysis helps researchers understand which factors have statistically meaningful conditional associations with salinity after accounting for all other measured variables.




&#x09;

&#x09;

&#x09;





&#x20;



&#x20;

&#x09;

&#x09;

&#x09;


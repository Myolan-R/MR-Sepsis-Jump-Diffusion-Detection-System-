# MR-Sepsis-Jump-Diffusion-Detection-System-
Sepsis Early Warning with Jump Diffusion Features

An in progress Python project that borrows jump diffusion ideas from quantitative finance and applies them to hourly ICU vital signs, to look for early signs of sudden patient deterioration such as sepsis.

This is a personal research and learning project. It is not a clinical tool and must not be used to make medical decisions.

The idea

In finance, jump diffusion models describe a price that drifts and wobbles most of the time but occasionally makes a sudden, large move. Patient vital signs look similar. Most hours show ordinary variation, but deterioration can appear as abrupt shifts. The project asks whether treating those shifts as "jumps", and measuring how often they happen and how vitals jump together, separates septic patients from non septic patients, and whether septic patients fall into distinct deterioration patterns.

Data
PhysioNet/Computing in Cardiology Challenge 2019, training Set A (about 20,300 patients, roughly 790,000 hourly rows).
Set B comes from a different hospital system and is held back as a final test set. It has not been used yet.
The data is not included in this repository. Download it from the PhysioNet challenge page and place it in a local data/ folder, which is ignored by git.

Method
Compile the data. Individual patient files are combined into one table (compile_sepsis_data.py).
Audit missingness. EtCO2 was 100% missing and was dropped. Temperature was about 66% missing and was kept with a wider 20 hour rolling window. Lab values were more than 90% missing, so they are used only as slow changing context through forward fill and a "hours since last update" feature, not for jump detection.

Detect jumps. Jump detection runs only on the eight vitals that are measured close to hourly. A jump is an hourly change larger than three standard deviations of that patient's own rolling variability.
Add trend features. Rolling trend features capture gradual deterioration, which jump detection alone would miss.
Count co moves. Co jump and co trend counts measure how often several vitals move abruptly in the same hour.
Compare cohorts. Covariance and correlation between vitals are compared for septic patients and matched non septic patients.
Cluster septic patients. K means with k equal to 3 is run on summary features, after removing patients with too many missing summary values.

Fit jump diffusion models. Fitting every patient separately failed, so parameters are now pooled by group (fit_jump_diffusion.py). See the results below.
Results so far
Septic patients split into three groups by volatility
Cluster	Patients	Mean heart rate	Heart rate standard deviation	Heart rate jumps per patient
0, high volatility	244	87.1	11.7	0.91
1, calm	581	87.3	6.5	0.06
2, moderate	604	88.4	9.9	0.28

Mean heart rate is almost identical across clusters. What separates them is variability and jump frequency, which averages would hide. The high volatility cluster has roughly fifteen times as many heart rate jumps per patient as the calm cluster. The same ordering appears for oxygen saturation and diastolic pressure.

Heart rate moves more closely with blood pressure in septic patients

Heart rate's correlation with systolic, mean and diastolic blood pressure is about 0.06 to 0.08 higher in septic patients than in matched non septic patients. This held up after I found and fixed a data leakage bug in how the non septic reference group was built.

Pooled jump rate estimates

Estimated jump rate (lambda) by group, where 0.5 is the upper limit of the fit.

Vital	Non septic	Cluster 0	Cluster 1	Cluster 2
Heart rate	0.32	0.26	0.30	0.30
MAP	0.18	0.14	0.19	0.22
Diastolic pressure	0.41	0.43	0.36	0.39
Oxygen saturation	0.21	0.21	0.50	0.22
Systolic pressure	0.50	0.50	0.41	0.50
Respiratory rate	0.50	0.50	0.50	0.50

Temperature models exist only for the non septic group (0.30) and cluster 2 (0.31), because of limited data.

What did not work, and what I changed

My first approach fitted a five parameter jump diffusion model to each patient separately. Over half of those fits (52.5%) hit the boundary on the jump rate, because an individual patient has too few observations to estimate jumps reliably.

I restructured the fitting into a hybrid. The jump rate, jump size mean and jump size spread are fitted once per group using pooled data, and only the drift and ordinary volatility are fitted per patient, which needs just two parameters. This produced 26 group level models and 112,078 patient level models.

Limitations
Clusters are not yet validated against outcomes. The clusters were built from jump and volatility features, so it is expected that they separate on those features. Whether they relate to what happens to patients is still untested.
Jump rate alone does not clearly separate septic from non septic patients. Heart rate jump rates are similar across groups. Any difference is more likely to be in jump size or in how vitals jump together, and I have not shown that yet.
Some estimates are unreliable. Respiratory rate and most systolic pressure estimates sit at the fit boundary, likely because those measurements are discrete or narrow in range. Heart rate and MAP are the most trustworthy.
No predictive performance yet. I have not reported accuracy, AUROC or lead time, and I do not claim that the features predict sepsis.
Next steps
Patient level train and test split, with a check that no patient appears in both.
Baseline classifier on raw vitals, then a gradient boosted model with the jump features added.
Evaluation using AUROC, recall at a fixed false positive rate, and lead time before sepsis onset. Accuracy is not used, because sepsis hours are rare.
Validate the clusters against outcomes.
Lead time test that truncates each held out septic patient's timeline at increasing distances before onset, and checks when the risk score starts to rise.
Final evaluation on Set B.
Sensitivity checks with 2 and 2.5 standard deviation thresholds and different label horizons.
Repository contents
compile_sepsis_data.py compiles raw patient files into one table.
Jump_Diffusion_Model.py contains the jump detection and feature code.
fit_jump_diffusion.py fits the pooled group models and per patient models.
A Jupyter notebook runs the analysis and produces the figures.
Running the code
Install Python 3 with numpy, pandas, scipy, scikit-learn and Jupyter.
Download the PhysioNet 2019 training data into data/.

Run compile_sepsis_data.py, then the notebook, then fit_jump_diffusion.py.
Data and attribution

This project uses the PhysioNet/Computing in Cardiology Challenge 2019 dataset, which contains hourly ICU records for patients from two hospital systems, with a label marking sepsis onset. The data is publicly available and is not included in this repository. You can download it from https://physionet.org/content/challenge-2019/1.0.0/ and place the training files in a local folder called data/, which is ignored by git. This project uses training Set A for development, and Set B is held back as a final test set. If you reuse the data, please follow the licence and citation requirements on the PhysioNet page.

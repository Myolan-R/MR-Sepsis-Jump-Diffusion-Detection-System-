Sepsis Early Warning with Jump Diffusion Features

An in progress Python project that borrows jump diffusion ideas from quantitative finance and applies them to hourly ICU vital signs, to look for early signs of sudden patient deterioration such as sepsis.

This is a personal research and learning project. It is not a clinical tool and must not be used to make medical decisions.

The idea

In finance, jump diffusion models describe a price that drifts and wobbles most of the time but occasionally makes a sudden, large move. Patient vital signs look similar. Most hours show ordinary variation, but deterioration can appear as abrupt shifts. The project asks whether treating those shifts as "jumps", and measuring how often they happen and how vitals jump together, separates septic patients from non septic patients, and whether septic patients fall into distinct deterioration patterns.

Data

Data comes from the PhysioNet/Computing in Cardiology Challenge 2019, available at https://physionet.org/content/challenge-2019/1.0.0/. The data is publicly available and is not included in this repository. Please follow the licence and citation requirements on the PhysioNet page if you reuse it.

Item	Value
Training set used	Set A
Patients	20,336
Hourly rows	790,215
Sepsis positive rows	17,136 (2.17% of rows)
Septic patients	1,790
Septic patients with history before onset	1,587
Septic patients clustered	1,429

Set B comes from a different hospital system and is held back as a final test set. It has not been used yet.

Method
Compile the data. Individual patient files are combined into one table (Compile_Sepsis_Data.py).
Audit missingness. EtCO2 was 100% missing and is not used. Temperature was about 66% missing and was kept with a wider 20 hour rolling window. Lab values were more than 90% missing, so they are not jump detected.

Detect jumps. Jump detection runs on seven vitals that are measured close to hourly after forward filling (heart rate, oxygen saturation, temperature, systolic, mean and diastolic pressure, and respiratory rate). A jump is an hourly change larger than three rolling standard deviations of that patient's own hourly changes, using a 12 hour window (20 hours for temperature).

Add trend features. Rolling trend features capture gradual deterioration, which jump detection alone would miss.
Compare cohorts. Correlations between hourly changes in each vital are compared for septic patients, using the 24 hours before sepsis onset, and non septic patients, using a matched 24 hour window. The matched reference point sits at the median septic onset position, which is 75.7% of the way through the stay.

Cluster septic patients. Each septic patient's history before onset is summarised, patients with too many missing summary values are removed, and k means with k equal to 3 is run on the scaled features (Clustering_Mechanics.py).
Fit pooled jump diffusion models. The jump rate, jump size mean and jump size spread are fitted once per group, using pooled data from the three septic clusters and the non septic cohort. Only the drift and ordinary volatility are fitted per patient (Fit_Jump_Diffusion.py and Jump_Diffusion_Model.py).

Results so far
Septic patients split into three groups by volatility
Cluster	Patients	Mean heart rate	Heart rate standard deviation	Heart rate jumps per patient
0, high volatility	244	87.1	11.7	0.91
1, calm	581	87.3	6.5	0.06
2, moderate	604	88.4	9.9	0.28

Mean heart rate is almost identical across clusters. What separates them is variability and jump frequency, which averages would hide. The high volatility cluster has roughly fifteen times as many heart rate jumps per patient as the calm cluster. The same ordering appears for oxygen saturation and diastolic pressure.

Heart rate moves more closely with blood pressure in septic patients

The correlation between hourly heart rate changes and hourly blood pressure changes is higher in septic patients than in the matched non septic group.

Vital pair	Septic correlation minus non septic correlation
Heart rate and systolic pressure	0.080
Heart rate and mean pressure	0.064
Heart rate and diastolic pressure	0.056

Differences for the other vital pairs are smaller, at 0.035 or less in size. These are small differences in correlation, so I treat them as a hint worth testing and not as a finding. The result held up after I found and fixed a data leakage bug in how the non septic reference group was built.

Pooled jump rate estimates

Estimated jump rate (lambda) by group, where 0.5 is the upper limit of the fit.

Vital	Non septic	Cluster 0	Cluster 1	Cluster 2
Heart rate	0.32	0.26	0.30	0.30
MAP	0.18	0.14	0.19	0.22
Diastolic pressure	0.41	0.43	0.36	0.39
Oxygen saturation	0.21	0.21	0.50	0.22
Systolic pressure	0.50	0.50	0.41	0.50
Respiratory rate	0.50	0.50	0.50	0.50

Temperature models exist only for the non septic group (0.30) and cluster 2 (0.31), because of limited data. Fitting produced 26 group level models and 112,078 patient level models.

What did not work, and what I changed

My first approach fitted a five parameter jump diffusion model to each patient separately. Of 103,740 patient and vital fits, about 41% hit the lower bound on the jump rate and about 12% hit the upper bound, so more than half were stuck at a boundary. An individual patient has too few jumps to estimate their rate reliably. The code for that earlier run is not included here.

I restructured the fitting into the pooled approach described above, so the jump parameters are estimated from many patients at once and only two parameters are fitted per patient.

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
Compile_Sepsis_Data.py combines the raw patient files into one table.
Clustering_Mechanics.py builds jump and trend features, summarises septic patients and clusters them.
Fit_Jump_Diffusion.py runs the pooled fitting across all groups and patients.
Jump_Diffusion_Model.py contains the jump diffusion likelihood, fitting and jump probability functions.
Sepsis_detection.ipynb is the exploration notebook, with plots for jump detection, the cohort comparison and cluster selection.
Running the code
Install Python 3 with pandas, numpy, scipy, scikit-learn, matplotlib, seaborn and Jupyter.
Download the PhysioNet 2019 training Set A and place the .psv files in data/training_setA/.
Run python Compile_Sepsis_Data.py.
Run python Clustering_Mechanics.py, or work through the notebook, to produce septic_cluster_assignments.csv.
Run python Fit_Jump_Diffusion.py. This is the slow step.

Generated files such as the combined CSV are ignored by git and are recreated by the scripts. the licence and citation requirements on the PhysioNet page.

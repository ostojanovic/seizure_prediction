# Seizure prediction using nonnegative matrix factorization

This repository contains code for the project "Seizure prediction using nonnegative matrix factorization". 

In this project, a procedure for the patient-specific prediction of epileptic seizures is developed by using a combination of [nonnegative matrix factorization][1] and smooth basis functions with robust regression, applied to power spectra of [intracranial electroencephalographic (iEEG)][2] signals. Linear support vector machines (SVM) with L1 regularization are used to select and weigh the contributions from different number of not equally informative channels among patients. Due to class imbalance in data, [synthetic minority over-sampling technique (SMOTE)][3] is applied. 

The data used in this project is a part of the [EPILEPSIAE][4] dataset, which is not publicly available and the [Epilepsyecosystem][5] dataset. For this reason, there are no specific paths or identification numbers in the code. The code for preprocessing the EPILEPSIAE dataset was developed in [MATLAB][6], while code for prediction and visualizations was developed in [Python3][7], as well as all code for the Epilepsyecosystem dataset. 

[1]: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
[2]: https://en.wikipedia.org/wiki/Electrocorticography
[3]: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
[4]: http://www.epilepsiae.eu/project_outputs/european_database_on_epilepsy
[5]: https://www.epilepsyecosystem.org/
[6]: https://ch.mathworks.com/de/products/matlab.html
[7]: https://www.python.org/

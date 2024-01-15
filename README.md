# ASD Machine Learning Diagnosis

Repository for research into diagnosing ASD (autism spectrum disorder) using machine learning.
This research has been written into a paper, available in ElasticRemurs.pdf.

Uses many resources from the PyKale library: https://github.com/pykale/pykale.

Pipeline adapted from: https://github.com/pykale/pykale/tree/main/examples/multisite_neuroimg_adapt.

## Areas of interest

The code in this repository:
* applies various methods of classification to diagnose ASD
* applies the Remurs[[1]](#remurs-ref) method to the classification task on the ABIDE dataset
* proposes Elastic-Remurs method, an extension of Remurs with an additional smoothing penalty

Future areas of investigation:
* treat the decision function output as a certainty measure rather than just output a class
* use phenotypic data in the classification using Remurs and Elastic-Remurs

## Usage
After cloning the repository, first install dependencies.
```
pip install -r requirements.txt
```

Everything will now be run from the code folder:
```
cd code
```

### Running the Classification Pipeline
Before running the classification pipeline, you can configure the pipeline by editing config.py.

After configuration, run the pipeline with:
```
python main.py
```

### Viewing results
To view results, use the view_results.py script. This script should also be run from the code directory.
```
python scripts/view_results.py results/{your_result_file}.csv
```
This basic view of the results will only show the alpha, beta, gamma values and the accuracy score. This will be updated.

## References
<a id="remurs-ref"></a>
[1] Song, X., & Lu, H. (2017). Multilinear Regression for Embedded Feature Selection with Application to fMRI Analysis. Proceedings of the AAAI Conference on Artificial Intelligence, 31(1). https://doi.org/10.1609/aaai.v31i1.10871

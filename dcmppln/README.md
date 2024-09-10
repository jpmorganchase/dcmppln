<!-- Run `./cp-pre-commit.sh` to set up precommit files
 -->
 
 
# Hybrid classical-quantum portfolio optimization using random matrix theory-based subproblem identification

The repository has three different modules. 1) Cleaning of correlation matrices using random matrix theory 2) Decomposition pipeline using community detection and = other clustering 3) Portfolio optimization objectives (V0, V1,V1.5, V2)
When we find communities, we have two options. One with the restriction on the number of elements in a cluster, the other without any such restrictions.


## Directory Structure

- **main.py** : Contains the Pipeline object and components of the pipline: Denoiser, Clustering, Filtering.
- **optimizer.py** : Contains the Optimizer objects

- **comparing_optimization_times**: Contains files related to comparing optimization times of running CPLEX optimization routines on full problem against the decomposition pipeline. 
- **create_portfolio_instance**: Scripts for creating portfolio instances. This includes generating returns (w or without log) and correlation and covariance matrices
- **modularity_maximization**: Resources for maximizing modularity. It includes many community detection algorithms , including modified versions, key ones: louvain_generalized, modularity_spectral_optimization
- **notebooks**: Jupyter notebooks or other interactive notebooks related to the project. (3 separate subfolders-- a) comparing objectives: Contains the jaccard similarity, difference and ratio in objectives, <br> b) thresholding and c) processing_time_comparision   )
- **out_of_sample**: Data and scripts related to out-of-sample testing. Only in the continuous setting but this has end-end sharpe ratio computations on present -- future splits of time series data. 
- **random_matrix_theory**: Resources and files related to random matrix theory tools such as marchenko pastur fits, splitting correlatoins into C1, C2 and C3. 
- **run_scripts** : Contains script example to run the pipeline
- **results** : Contains notebooks and results csv to visualize the results
- **thresholding**: Scripts and data for thresholding techniques. Also, includes comparision of thresholded objectives (it is in utils/utils.py)
- **utils**: Utility scripts and helper functions for the project. (Need to add a separate readme file)
- **others**: Miscellaneous files and resources not categorized elsewhere. This includes integrality gap experiments

- **data**: This is an empty folder as a placeholder for the data files, data files should be as follows 
  - self.path+"_correlation.npy"
  - self.path+"_covariance.npy"
  - self.path+"_returns.npy"


## Getting Started

1. **Setup environement with cplex**
```
cd ..
git clone git clone --depth=1 --no-single-branch https://bitbucketdc.jpmchase.net/scm/quantumalgorithms/environment.git
environment/scripts/cplex_install.sh
source py38_cplex/bin/activate
cd atithi_acharya_summer_2023_internship_project/
pip install -r requirement_pierre.txt
```

2. **Usage**: Briefly explain how to use the project or run the scripts. <br>
On a new omni-ai instance
If you do not want to create an environment for docplex
Follow the instructions to setup packages in 1 <br>
a) git clone https://bitbucketdc.jpmchase.net/scm/quantumalgorithms/atithi_acharya_summer_2023_internship_project.git
b) The data folder is empty, when the user runs create_portfolio_instance (including notebook) , they have to save the files there in order to run.
c) get data and results stored on aws s3
```
aws s3 cp --recursive s3://app-id-108383-dep-id-105006-uu-id-ahc5pxjsqas8/decomposition_pipeline/ .
```

## Contributing

Provide guidelines on how others can contribute to your project.

## License

Add license under (ask)

## Acknowledgements

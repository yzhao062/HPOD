
**Supplementary Material**: HPOD: Hyperparameter Optimization for Unsupervised Outlier Detection

----

To run the demo for the three OD algorithms, first install the required libraries by executing the following command:
"pip install -r requirements.txt". It is working with Python 3.7+. Our experiment is in Python 3.9.

To run the demo for RAE, execute:
"python demo_hpod_rae.py".

Similarly, to run demo for LOF, execute:
"python demo_hpod_lof.py".

Also, to run demo for iForest, execute:
"python demo_hpod_iforest.py".

More file description:
- init_meta.py includes the implementation of meta-initialization
- utility.py includes a set of helper functions.
- models folder includes pre-trained models for fast replication.
- datasets folder includes the raw file of all datasets.

If you face any execution issue, please feel free to open an issue (anonmously if you are a reviewer.)

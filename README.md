# SVCMEM

This SVCMEM package is developed by Yue Shan, Chao Huang, and Hongtu Zhu from the [BIG-S2 lab](https://www.med.unc.edu/bigs2/).

SVCMEM (Spatially Varying Coefficient Mixed Effects Model) is a R/Python mixed coding based package for evaluating ensemble learning in multiple neuroimaging studies.
The aim of this package is to  systematically investigate merging and ensembling methods for spatially varying coefficient mixed effects models
(SVCMEM) in order to carry out  integrative learning of  neuroimaging data obtained from multiple  biomedical  studies. 
The "merged" approach involves training a single learning model using a comprehensive dataset that encompasses information from all the studies. Conversely, the "ensemble" approach involves creating a weighted average of distinct learning models, each developed from an individual study. 
We systematically investigate the prediction accuracy of the merged and ensemble learners under the presence of different degrees of inter-study heterogeneity.  
Additionally, we establish asymptotic guidelines for making strategic decisions about when to employ either of these models in different scenarios, along with deriving optimal weights for the ensemble learner. 
To validate our theoretical results, we perform extensive simulation studies. 
# Command Script

Pipeline: run the following scripts in order:
0.preparation
  0.1 modify PATH in the following 3 scripts into your path containing all scripts contained in this package
  0.2 create the file directory structure (as in section B)
1.generate simulation data [gen_dat.R]
  output:
  - x.txt
    [covariate matrix from multiple studies]
  - y.txt
    [imaging responses from multiple studies]
  - beta.txt
    [true varying coefficients]
2.run both merged learning and ensemble learning [mvcm.py]
  output:
  - MSPE.txt
    [mean square prediction error for both merged learning and ensemble learning]
3.calculate the decision point [thr.R]
  output:
    thr_sig2_uneq.txt
    [three strategy decision points in the theoretical analysis]
    

# Robust Least Squares and Outlier Detection
Fitting a known model robustly to data using bayesian iteration. The two implementations use
* RANSAC
* M-Estimates

The robust part is implemented, fitting the function is not. Model 
fitting is borrowed from the scipy.minimize. Feel free to use a different model fitting method.  

## Pre-requisites
**numpy** is the only pre-requisite for **robust_lsq.py**.
**robust_lsq.py** requires a least squares fitting function
(or some other fitting function), 
such as **scipy.optimize.minimize**. Please see example
**models.py**. 

### robust_lsq.py
* numpy

### models.py
* scipy
* numpy

### test.py
* scipy
* numpy
* matplotlib

## Setup
Please run **test.py** for an example of fitting a straight line
to data robustly with bayesian estimation. 

## How does it work? 
The key idea is to determine the samples that fit the model best. 
Bayesian updates are used. Bayes rule is given by: 

P(data/model) = P(model/data)*P(data)/p(model)

P(data/model) := normalization(P(model/data)*P(data))

Note:
1. P(model) is a constant and can be ignored. 
1. In the next iteration P(data/model) becomes P(data).

### ALGORITHM
From an implementation perspective, these are the steps: 
1. Build P(data) uniform distribution (or with prior knowledge) over data. 
1. Sample n samples from data distribution. 
1. Fit model to the selected n samples. 
Essentially we are selecting(sampling) the best model given the data. 
This is the P(model/data) step. 
1. Estimate a probability distribution: P(data/model). 
    1. These are the errors of the data given the selected model. 
    1. It is wise to use a function such as arctan(1/errors)
    so errors are not amplified and create a useless probability distribution.
    
1. Compute P(data) with update: P(data/model) = normalize(P(data/model)*P(data)) 
    1. Normalize probability distribution. 
    1. This is the bayesian update step. 
    
1. Go to step 2. and iterate until desired convergence of P(data).  


### RANSAC
For a RANSAC flavor of bayesian robust fitting, k samples are selected to fit the model.
#### In classical RANSAC:
1. The minimum number of samples (k) to fit a model is used.
1. k samples are randomly selected p times.  
1. The best set in p of k samples that fit all the data is selected.   

#### In this bayesian flavor:
1. k samples are selected and fit using least squares (or something else). 
1. Samples are selected from a probability distribution estimated using bayesian updates. 

### M-Estimates
This is similar to RANSAC except when fitting the model, all samples are used to
fit the model but are weighed according to their probability distribution. 
The probability distribution(weights) is updated using bayesian updates. 
 
### Outlier detection
 The probability distribution over the data P(data) provides a way to 
 perform outlier detection. Simply apply a threshold over this distribution. 


## License
Copyright 2018 Guru Subramani

Permission is hereby granted, free of charge, 
to any person obtaining a copy of this software and 
associated documentation files (the "Software"), 
to deal in the Software without restriction, 
including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, 
and to permit persons to whom the Software 
is furnished to do so, subject to the following 
conditions:

The above copyright notice and this permission 
notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", 
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS 
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
USE OR OTHER DEALINGS IN THE SOFTWARE.



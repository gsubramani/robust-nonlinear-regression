# Robust Least Squares
Fitting a known model robustly to data. The two implementations use
* RANSAC
* M-Estimates

The robust part is implemented, fitting the function is not. Model 
fitting is borrowed from the scipy.minimize. 

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
to data robustly. 

## How does it work? 
### RANSAC


## license
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



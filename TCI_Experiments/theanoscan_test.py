# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/Users/xinghualu/.spyder2/.temp.py
"""
from theano import function
import theano.tensor as T
import theano
import numpy as np
import math


########### the following block define logSum function 
#  calcuate log( x + y ) given log(x) and log(y)
maxExp = -4950.0 
lnX,lnY = T.scalars(2)

yMinusx =lnY - lnX  ## this part is for the condition which lnX >lnY
xMinusy = lnX -lnY  # if lnX <lnY
bigger = T.switch(T.gt(lnX,lnY), lnX, lnY)
YSubtractX = T.switch(T.gt(lnX, lnY), yMinusx, xMinusy)       
 
x_prime =  T.log(1 + T.exp(YSubtractX)) + bigger
calcSum = T.switch(T.lt(YSubtractX, maxExp), bigger, x_prime)
logSum = function([lnX,lnY], calcSum, allow_input_downcast=True)

##############################################################

def logSumTheano(lnX, lnY  ):
    return calcSum

# now try to calculate the accummulate logSum over a vector using scan
seq = T.vector('seq')   
outputs_info = T.as_tensor_variable(np.asarray(-18.0, seq.dtype))  # close to log (0)

scan_result, scan_updates = theano.scan(fn = logSumTheano,
                                        outputs_info=outputs_info,
                                        sequences=seq)
#final_result = scan_result[-1]
logAccumSum = theano.function(inputs=[seq], outputs=scan_result)

logSeq = np.asarray([math.log(2), math.log(3), math.log(4)])

print logAccumSum(logSeq)

print math.log(9)


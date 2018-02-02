# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 17:36:04 2015

@author: jr
"""

from gpa.gpa import GlobalPatternAnalysis
import pickle
import warnings
import re

warnings.filterwarnings("ignore")


# load data
tf = open('../../../../4_data/moby_dick_ch1-10.txt', 'r')
data = re.sub(r"\s+", "", tf.read())
print(data)
exit(-1)

# learn vector space
lbs = GlobalPatternAnalysis(data)

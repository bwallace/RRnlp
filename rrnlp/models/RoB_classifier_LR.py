"""
the BiasAbRobot class takes the *abstract* of a clinical trial as
input as a string, and returns bias information as a dict which
can be easily converted to JSON.

V2.0

Returns an indicative probability that the article is at low risk of bias, based on the abstract alone.




"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@northeastern.edu>

import json
import uuid
import os
import rrnlp
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import numpy as np
import re
import scipy
from scipy.sparse import hstack
import rrnlp


weights_path = rrnlp.models.weights_path

doi = rrnlp.models.files_needed['RoB_classifier_LR']['zenodo']

class AbsRoBBot:
    ''' Lightweight container class that holds RoB logistic regression model '''

    def __init__(self):
        
        with open(os.path.join(weights_path, f'{doi}_bias_prob_clf.pck'), 'rb') as f:
            self.clf = pickle.load(f)

        self.vec = HashingVectorizer(ngram_range=(1, 3), stop_words='english')


    def predict_for_doc(self, ti_and_abs: str) -> dict: 

        """
        Annotate abstract of clinical trial report
            
        """
        X = self.vec.transform([ti_and_abs])

        probs = self.clf.predict_proba(X)[:,1].tolist()
        
        return {"prob_low_rob": probs[0]}

#!/usr/bin/env python
# coding: utf-8

import xlearn as xl
import pandas as pd

fm_model = xl.create_fm()  
# fm_model = xl.create_linear()            
fm_model.setTrain("./data/index_first_training.csv") 
fm_model.setValidate("./data/index_first_testing.csv")
fm_model.disableEarlyStop();



param = {'task':'binary', 'lr':0.02, 'lambda':0.002, 'metric': 'auc', 'epoch':20}

fm_model.fit(param, "./fm_model.out")

fm_model.setSign()
fm_model.setTest("./data/index_first_testing.csv")
fm_model.predict("./fm_model.out", "./output.txt")






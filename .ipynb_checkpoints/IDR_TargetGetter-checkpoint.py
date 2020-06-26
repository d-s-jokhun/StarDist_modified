#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Identifies target from the IDR database according to CompoundsOfInterest


# In[1]:


import os
import requests
import csv


# In[6]:


def IDR_TargetGetter(CompoundsOfInterest, idx_file=None):
    
    if idx_file is None:
        url = 'https://raw.githubusercontent.com/d-s-jokhun/idr-metadata/master/idr0016-wawer-bioactivecompoundprofiling/screenA/idr0016-screenA-annotation.csv'
        idx_file = str(os.path.basename(url))
        r = requests.get(url)
        with open(idx_file, 'w') as idxfile:
            idxfile.write(r.text)
    
    Targets=[]
    with open(idx_file, newline='') as idxfile:
        idx_reader = csv.DictReader(idxfile,fieldnames=None, restkey=None, restval=None, dialect='excel')
        for row in idx_reader:
            if CompoundsOfInterest.count(row['Compound Name']) > 0 :
                Targets.append({'Plate':row['Plate'],
                               'Well':row['Well'],
                               'Compound Name':row['Compound Name']})
    
    
    return Targets
#  targets = [{plate, Well, Compound}]


# In[ ]:





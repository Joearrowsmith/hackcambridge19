# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:14:28 2019

@author: Joonsu
"""
import re

def clean_data(text):
    #strip HTML and non-ASCII characters
    text = re.sub('<[^<]+?>', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text) 
    
    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')
    
    # Convert all letters to lowercase
    text = text.lower()
    
    return text


# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:23:01 2022

@author: Debabrata Ghorai, Ph.D.
"""

import datetime
import string
import random

ddmmyyyy = datetime.datetime.now().strftime("%d%m%y")
hhmmss = datetime.datetime.now().strftime("%H%M%S")
alphanumeric = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
wkt_id = ddmmyyyy+"_"+hhmmss+"_"+alphanumeric

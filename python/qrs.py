import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import wfdb

# Demo 16 - List the Physiobank Databases

dbs = wfdb.get_dbs()
display(dbs)
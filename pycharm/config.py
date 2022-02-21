
import pandas as pd
import matplotlib.pyplot as plt

# Display pandas df without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Globally setting font sizes via rcParams should be done with
'''
# paper params
params = {
    'font.size': 18,
    'figure.figsize': (7,3.5),
    'figure.dpi': 80,
    'savefig.dpi': 300,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.labelsize': 11,
    'axes.axisbelow': True
         }
'''

# test params
params = {
    'font.size': 18,
    'figure.figsize': (15, 5),
    'figure.dpi': 160,
    'savefig.dpi': 300,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.labelsize': 11,
    'axes.axisbelow': True
}

plt.rcParams.update(params)

# The defaults can be restored using
# plt.rcParams.update(plt.rcParamsDefault)


# ARK —> ARGoS coordinate system
ARENA_SIZE = 1875
M_TO_PIXEL = 20
CM_TO_M = 100




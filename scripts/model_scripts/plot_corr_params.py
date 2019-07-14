import talos as ta
from talos.commands.reporting import Reporting
import sys 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = Reporting(sys.argv[1])
save_fname = sys.argv[2]

log.plot_corr(metric='val_fmeasure_acc')
plt.tight_layout()
plt.savefig(save_fname)

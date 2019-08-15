import sys
import pickle
import scipy.io as sio


fname = str(sys.argv[1])

Data = pickle.load(open(fname, 'rb'))
sio.savemat(fname.replace('pckl', 'mat'), {'data': Data})

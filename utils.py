import numpy as np 
import logging 
from Dynamics.Eco import Eco
from Dynamics.Gene import Gene 
from Dynamics.Epidemic import Epidemic 
from Dynamics.Neural import Neural  
 
 
def create_net(args):
    n = args.n
    avg_d = args.k
    if args.dynamics == 'eco':
        net = Eco(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
    elif args.dynamics == 'gene':
        net = Gene(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
    elif args.dynamics == 'epi':
        net = Epidemic(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed)
    elif args.dynamics == "wc":
        net = Neural(n, avg_d/(n-1), args.data, args.topology, m=args.m, seed=args.seed) 
     
    gt = read_param(args)
    net.gt = gt 
    net.obs_idx = list(range(net.N))
    return net
 

def read_param(args): 
    # obtain a common parameter
    if args.dynamics == 'gene': 
        gt = np.array([1,1,2])
    elif args.dynamics == 'eco': 
        gt = np.array([0.1, 5, 1, 5, 0.9, 0.1])
    elif args.dynamics == 'epi': 
        gt = np.array([0.5])
    elif args.dynamics == 'wc':
        gt = np.array([1,1]) 
    gt = np.array(gt, dtype=np.float64)
    return gt 

def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger
 
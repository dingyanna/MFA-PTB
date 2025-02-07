 
from utils import *  
from huge_net import * 
from er_density import * 
from sf_heterogeneity import *
from network_size import * 
import argparse

parser = argparse.ArgumentParser('Steady State Computation') 
parser.add_argument('--experiment', type=str, default='main', help='partial, noise, nobserve, nblock_bsize, adv, het')
parser.add_argument('--dynamics', type=str, default='eco', help="eco, epi, gene, wc, popu")
parser.add_argument('--topology', type=str, default='er', help="er, er")
parser.add_argument('--data', type=str, default='') 
# Topology related
parser.add_argument('--n', type=int, default=200) # Total number of nodes
parser.add_argument('--k', type=int, default=12) # average degree (parameter for ER network)
parser.add_argument('--m', type=int, default=4) # parameter for scale free network

parser.add_argument('--h', type=float, default=1e-4) # RK step size
parser.add_argument('--full_max_iter', type=float, default=2000) # RK num iter
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--avg_operator', type=str, default='avg')
  
 
if __name__ == '__main__':
    args = parser.parse_args()  
    globals()[args.experiment](args)
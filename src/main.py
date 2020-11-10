#!/usr/bin/python3
from scipy.constants import G as scipy_G

from interactions import *
import utils
import schemas


"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dt", help="delta t for calculating steps",
                    type=float)
parser.add_argument("target", help="target time to reach in the simulation",
                    type=int)
parser.add_argument("method", help="method to use for running the simulation",
                    choices=['eul1', 'eul2'])
parser.add_argument("-p", "--nparts", help="number of spark partitions for cartesian operations",
                    default=16, type=int)
parser.add_argument("-l", "--limit", help="limit the number of input rows to read",
                    nargs="?", const=1000, type=int)
parser.add_argument("-o", "--outputDir", help="output path",
                    default="../output/")
parser.add_argument("-i", "--inputDir", help="input path",
                    default="../data/")
parser.add_argument("-G", help="gravitational constant for the simulation",
                    default=scipy_G, type=float)
parser.add_argument('--test', help="run test.py instead of simulation",
                    action='store_true')
args = parser.parse_args()
"""/arguments"""


df_t0 = utils.load_df("c_0000.csv", args.inputDir,
                      schema=schemas.clust_input, part="id", limit=args.limit)

methods = {
    "eul1": Intergrator_Euler(args.dt, args.nparts, args.G),
    "eul2": Intergrator_Euler_Sym(args.dt, args.nparts, args.G),
}

sim = Simulation(df_t0, methods[args.method], args.target, args.G)

sim.run()

utils.save_df(sim.cluster, f"c_{sim.t}_dt_{args.dt}", args.outputDir)

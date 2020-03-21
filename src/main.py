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
parser.add_argument("--log", help="limit the number of input rows to read",
                    default="0", choices=[0, 1, 2, 3, 4], type=int)
args = parser.parse_args()
"""/arguments"""


df_t0 = utils.load_df("c_0000.csv", args.inputDir,
                      schema=schemas.clust_input, part="id", limit=args.limit)


sim = Simulation(df_t0, args.dt, args.target, args.nparts, args.G)

if args.log > 0:
    from log import ClusterLogDecorator
    from pyspark.context import SparkContext

    sc = SparkContext.getOrCreate()
    clogger = ClusterLogDecorator(
        "clusterLogger", f"log-{utils.clean_str(sc.appName)}-{sc.applicationId}.csv",
        args.outputDir, "header", False, True, args.log, args.G)

    sim.calc_F = clogger(sim.calc_F)
    #get the final state as well, albeit without F
    sim.run = clogger(sim.run)

sim.run(args.method)

utils.save_df(sim.cluster, f"c_{args.target:04d}_dt_{args.dt}", args.outputDir)

#!/usr/bin/python3
from scipy.constants import G as scipy_G

from interactions import *
import utils
import schemas
from pyspark.sql.session import SparkSession

"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dt", help="delta t for calculating steps",
                    type=float)
parser.add_argument("target", help="target time to reach in the simulation",
                    type=float)
parser.add_argument("method", help="method to use for running the simulation",
                    choices=['eul1', 'eul2', 'rk4'])

parser.add_argument("-n", help="number input particles (for tuning)",
                    default=64000, type=int)

parser.add_argument("--dtout", help="time interval between cluster snapshots",
                    default=None, type=float)
parser.add_argument("--dtdiag", help="time interval between cdiagnosting output",
                    default=None, type=float)

parser.add_argument("-p", "--nparts", help="number of spark partitions for cartesian operations",
                    default=4, type=int)
parser.add_argument("-l", "--limit", help="limit the number of input rows to read",
                    nargs="?", const=1000, type=int)

parser.add_argument("-o", "--outputDir", help="output path",
                    default="../output/")
parser.add_argument("-i", "--input", help="input",
                    default="../data/")


parser.add_argument("-G", help="gravitational constant for the simulation",
                    default=scipy_G, type=float)

args = parser.parse_args()
"""/arguments"""


"""load data"""


"""adjust spark settings"""
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.caseSensitive", "true")
if args.n < 8:
    spark.conf.set("spark.default.parallelism", "1")
    spark.conf.set("spark.sql.shuffle.partitions", "1")
    df_t0 = utils.load_df(args.input,
                          schema=schemas.clust, part=1, limit=args.limit)
else:
    df_t0 = utils.load_df(args.input,
                          schema=schemas.clust, part="id", limit=args.limit)


"""setup simulation"""
methods = {
    "eul1": Intergrator_Euler(args.dt, args.nparts, args.G),
    "eul2": Intergrator_Euler2(args.dt, args.nparts, args.G),
    "rk4": Integrator_RungeKutta4(args.dt, args.nparts, args.G),
}

sopts = utils.SaveOptions(args.outputDir, fformat="csv", compression="none", header="true")
sim = Simulation(df_t0, methods[args.method], args.target, sopts,
                 dt_out=args.dtout, dt_diag=args.dtdiag)


"""run"""
sim.run()

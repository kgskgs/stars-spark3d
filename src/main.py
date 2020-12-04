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
                    choices=['eul1', 'eul2'])

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
parser.add_argument("-i", "--inputDir", help="input path",
                    default="../data/")
parser.add_argument("-f", "--inputFile", help="input filename",
                    default="c_0000.csv")

parser.add_argument("-G", help="gravitational constant for the simulation",
                    default=scipy_G, type=float)

parser.add_argument('--test', help="run test.py instead of simulation",
                    action='store_true')
args = parser.parse_args()
"""/arguments"""


"""load data"""
df_t0 = utils.load_df(args.inputFile, args.inputDir,
                      schema=schemas.clust, part="id", limit=args.limit)


"""adjust spark settings"""
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.caseSensitive", "true")
if df_t0.count() < 4:
    spark.conf.set("spark.default.parallelism", "2")
    spark.conf.set("spark.sql.shuffle.partitions", "2")


"""setup simulation"""
methods = {
    "eul1": Intergrator_Euler(args.dt, args.nparts, args.G),
    "eul2": Intergrator_Euler2(args.dt, args.nparts, args.G),
}

sopts = utils.SaveOptions(args.outputDir, fformat="csv", compression="None", header="true")
sim = Simulation(df_t0, methods[args.method], args.target, sopts,
                 dt_out=args.dtout, dt_diag=args.dtdiag)



"""run"""
sim.run()

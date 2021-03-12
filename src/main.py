#!/usr/bin/python3
from scipy.constants import G as scipy_G

from simulation import *
from integrators import *
import utils
import schemas
from pyspark.sql.session import SparkSession
import os

"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dt", help="delta t for calculating steps",
                    type=float)
parser.add_argument("target", help="target time to reach in the simulation",
                    type=float)
parser.add_argument("integrator", help="integrator to use for running the simulation",
                    choices=['eul1', 'eul2', 'rk4'])

parser.add_argument("--dtout", help="time interval between cluster snapshots",
                    default=None, type=float)
parser.add_argument("--dtdiag", help="time interval between cdiagnosting output",
                    default=None, type=float)

parser.add_argument("-l", "--limit", help="limit the number of input rows to read",
                    nargs="?", const=1000, type=int)

parser.add_argument("-o", "--outputDir", help="output path",
                    default="../output/")
parser.add_argument("-i", "--input", help="path(s) to input data")


parser.add_argument("-G", help="gravitational constant for the simulation",
                    default=scipy_G, type=float)

args = parser.parse_args()
"""/arguments"""

"""adjust spark settings"""
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.caseSensitive", "true")

"""load data"""
df_t0 = utils.load_df(args.input,
                      schema=schemas.clust, part="id", limit=args.limit)

"""setup simulation"""
methods = {
    "eul1": IntergratorEuler(args.dt, args.G),
    "eul2": IntergratorEuler2(args.dt, args.G),
    "rk4": IntegratorRungeKutta4(args.dt, args.G),
}

nameStr = utils.clean_str(spark.conf.get("spark.app.name")) + "-" + spark.conf.get("spark.app.id")
sopts = utils.SaveOptions(os.path.join(args.outputDir, nameStr), fformat="csv",
                          compression="none", header="true")

sim = Simulation(df_t0, methods[args.integrator], args.target, sopts,
                 dt_out=args.dtout, dt_diag=args.dtdiag)


"""run"""
sim.run()

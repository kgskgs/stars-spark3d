#!/usr/bin/python3

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
                    choices=['eul1', 'eul2', 'rk4', 'vlf'])
parser.add_argument("input", help="path(s) to input data")

parser.add_argument("--dtout", help="time interval between cluster snapshots",
                    default=None, type=float)
parser.add_argument("--dtdiag", help="time interval between cdiagnosting output",
                    default=None, type=float)
parser.add_argument("--saveDiag", help="should diagnostic be saved to disk instead of printed",
                    nargs="?", const=True, default=False, type=bool)
parser.add_argument("--addT", help="should t be added to cluster snapshots",
                    nargs="?", const=True, default=False, type=bool)

parser.add_argument("-l", "--limit", help="limit the number of input rows to read",
                    nargs="?", const=1000, type=int)

parser.add_argument("-o", "--outputDir", help="output path",
                    default="../output/")
parser.add_argument("-f", help="format to save output in",
                    choices=['parquet', 'csv'], default="parquet")
parser.add_argument("--comp", help="format to save output in",
                    type=str, default=None)


parser.add_argument("-G", help="gravitational constant for the simulation",
                    default=1, type=float)

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
    "vlf": IntegratorLeapfrog(args.dt, args.G),
}

nameStr = utils.clean_str(spark.conf.get("spark.app.name")) + "-" + spark.conf.get("spark.app.id")
sopts = utils.SaveOptions(os.path.join(args.outputDir, nameStr), fformat=args.f,
                          compression=args.comp, header="true")

sim = Simulation(df_t0, methods[args.integrator], args.target, sopts,
                 add_t_snap=args.addT, dt_out=args.dtout, dt_diag=args.dtdiag, saveDiag=args.saveDiag)


"""run"""
sim.run()

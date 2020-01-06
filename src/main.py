#!/usr/bin/python3
from scipy.constants import G as scipy_G

from interactions import *
import utils
import schemas


"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", help="number of input rows to read",
                    nargs='?', const=1000, type=int)
parser.add_argument("--outputDir", help="output path",
                    nargs='?', default="../output/")
parser.add_argument("--inputDir", help="input path",
                    nargs='?', default="../data/")
parser.add_argument("-G", help="gravitational constant for the simulation",
                    nargs='?', default=scipy_G, type=float)
args = parser.parse_args()
"""/arguments"""


df_in = utils.load_df("c_0000.csv", args.inputDir,
                      schema=schemas.clust_input, part="id", limit=args.limit)


df_F = calc_F(df_in, args.G)

utils.save_df(df_F, "F", args.outputDir)

"""

df_gforce_cartesian = calc_gforce_cartesian(df_in, args.G)

# get rid of reciprocal relationships when saving
pth = utils.save_df(df_gforce_cartesian.filter("id < id_other"),
                    "gforce_cartesian", args.outputDir)

df_gforce = utils.df_agg_sum(
    df_gforce_cartesian, "id", "gforce", "gx", "gy", "gz")

utils.save_df(df_gforce, "gforce_sum", args.outputDir)
"""
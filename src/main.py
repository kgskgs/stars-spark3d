#!/usr/bin/python3
from scipy.constants import G as scipy_G

import gravity
import utils


"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", help="number of input rows to read", nargs='?', const=1000, type=int)
parser.add_argument("--outputDir", help="output path", nargs='?', default="../output/")
parser.add_argument("--inputDir", help="input path", nargs='?', default="../data/")
parser.add_argument("-G", help="gravitational constant for the simulation", nargs='?', default=scipy_G , type=float)
args = parser.parse_args()
"""/arguments"""


df_gforce_cartesian = gravity.calc_gforce_cartesian("c_0000.csv", args.inputDir, args.limit, args.G)

#get rid of reciprocal relationships when saving
pth = utils.save_df(df_gforce_cartesian.filter("id < id_other"), 
	"gforce_cartesian", args.outputDir)

df_gforce = gravity.sum_gforce(df_gforce_cartesian) #"*.parquet", pth

utils.save_df(df_gforce, "gforce_sum", args.outputDir)
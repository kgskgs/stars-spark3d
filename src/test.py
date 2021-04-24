import cluster
import utils
import schemas
from pyspark.context import SparkContext
import os

"""arguments"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outputDir", help="output path",
                    default="../output/")
parser.add_argument("-i", "--input", help="input path",
                    default="../data/cluster/")
args = parser.parse_args()
"""/arguments"""

G = 1
TOLERANCE = 1e-04
res = []
data = ['c_0500.csv', 'c_0700.csv', 'c_0600.csv', 'c_1000.csv', 'c_0900.csv', 'c_1200.csv', 'c_1100.csv', 'c_1500.csv', 'c_0300.csv', 'c_1800.csv', 'c_1300.csv', 'c_0800.csv', 'c_1700.csv', 'c_0200.csv', 'c_0100.csv', 'c_0400.csv', 'c_0000.csv', 'c_1600.csv', 'c_1400.csv']

for fname in data:
    df = utils.load_df(os.path.join(args.input, fname),
                       schema=schemas.clust, part="id")
    e = cluster.calc_E(df)
    diff = abs(e - (-0.25))
    res.append([fname,
                e,
                -0.25,
                diff,
                ])

sc = SparkContext.getOrCreate()
res = sc.parallelize(res).toDF(schema=schemas.E_test_res)

utils.save_df(res, "E_TEST", args.outputDir, fformat="csv")

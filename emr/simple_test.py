import numpy as np

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pyspark.sql.functions as f

from math import sqrt
from scipy.constants import G

import os


"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", help="number of input rows to read", nargs='?', const=1000, type=int)
parser.add_argument("--outputDir", help="output path", nargs='?', default="../output/")
parser.add_argument("--inputDir", help="input path", nargs='?', default="../data/")
args = parser.parse_args()
"""/arguments"""

print("test")

spark = SparkSession.builder.getOrCreate()

def load_cluster_data(fname, pth=None, header="true", limit=None):
    """load a cluster from the dataset
    https://www.kaggle.com/mariopasquato/star-cluster-simulations 
    """
    schm = StructType([StructField('x', DoubleType(), True),
                        StructField('y', DoubleType(), True),
                        StructField('z', DoubleType(), True),
                        StructField('vx', DoubleType(), True),
                        StructField('vy', DoubleType(), True),
                        StructField('vz', DoubleType(), True),
                        StructField('m', DoubleType(), True),
                        StructField('id', IntegerType(), True)])

    if pth:
        floc = os.path.join(pth, fname)
    else: #default data location
        floc = os.path.join(os.path.dirname(__file__), '..', 'data', fname)

    df = spark.read.load(floc, 
                                format="csv", header=header, schema=schm)

    if limit:
        return df.limit(limit)
    return df

clust = load_cluster_data("c_0000.csv", pth=args.inputDir, limit=8000)

print("loaded")
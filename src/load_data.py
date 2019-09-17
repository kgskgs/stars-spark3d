from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

from pyspark3d.repartitioning import prePartition
from pyspark3d.repartitioning import repartitionByCol

sc = SparkContext('local')
spark = SparkSession(sc)


schm = StructType([StructField('x', DoubleType(), True),
                   StructField('y', DoubleType(), True),
                   StructField('z', DoubleType(), True),
                   StructField('vx', DoubleType(), True),
                   StructField('vy', DoubleType(), True),
                   StructField('vz', DoubleType(), True),
                   StructField('m', DoubleType(), True),
                   StructField('id', IntegerType(), True)])


df = spark.read.load("../data/c_0000.csv", format="csv", header="true", schema=schm)

df.show(5)

options = {
    "geometry": "points",
    "colnames": "x,y,z",
    "coordSys": "cartesian",
    "gridtype": "onion"}


df_colid = prePartition(df, options, numPartitions=20)

df_colid.show(5)

df_repart = repartitionByCol(df_colid, "partition_id", preLabeled=True, numPartitions=20)

df_repart.show(5)


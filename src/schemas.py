#!/usr/bin/python3
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT

clust_input = StructType([
    StructField('x', DoubleType(), False),
    StructField('y', DoubleType(), False),
    StructField('z', DoubleType(), False),
    StructField('vx', DoubleType(), False),
    StructField('vy', DoubleType(), False),
    StructField('vz', DoubleType(), False),
    StructField('m', DoubleType(), False),
    StructField('id', IntegerType(), False)
])

"""SPLIT"""
dist_gforce = StructType([
    StructField("dist", DoubleType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gx", DoubleType(), False),
    StructField("gy", DoubleType(), False),
    StructField("gz", DoubleType(), False)
])

dist_gforce_cartesian = StructType([
    StructField('id', IntegerType(), False),
    StructField('id_other', IntegerType(), False),
    StructField("dist", DoubleType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gx", DoubleType(), False),
    StructField("gy", DoubleType(), False),
    StructField("gz", DoubleType(), False)
])

gforce_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gx", DoubleType(), False),
    StructField("gy", DoubleType(), False),
    StructField("gz", DoubleType(), False)
])


F = StructType([
    StructField("Fx", DoubleType(), False),
    StructField("Fy", DoubleType(), False),
    StructField("Fz", DoubleType(), False)
])

F_cartesian = StructType([
    StructField('id', IntegerType(), False),
    StructField('id_other', IntegerType(), False),
    StructField("Fx", DoubleType(), False),
    StructField("Fy", DoubleType(), False),
    StructField("Fz", DoubleType(), False)
])


F_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("Fx", DoubleType(), False),
    StructField("Fy", DoubleType(), False),
    StructField("Fz", DoubleType(), False)
])


r_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("x", DoubleType(), False),
    StructField("y", DoubleType(), False),
    StructField("z", DoubleType(), False)
])

v_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("vx", DoubleType(), False),
    StructField("vy", DoubleType(), False),
    StructField("vz", DoubleType(), False)
])

"""VECTORS"""
v_dist_gforce = StructType([
    StructField("dist", DoubleType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gv", VectorUDT(), False),
])

v_dist_gforce_cartesian = StructType([
    StructField('id', IntegerType(), False),
    StructField('id_other', IntegerType(), False),
    StructField("dist", DoubleType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("g", VectorUDT(), False),
])

v_gforce_effective = StructType([
    StructField('id', IntegerType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gv", VectorUDT(), False)
])

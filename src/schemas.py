#!/usr/bin/python3
from pyspark.sql.types import *

clust = StructType([
    StructField('id', IntegerType(), False),
    StructField('x',  DoubleType(), False),
    StructField('y',  DoubleType(), False),
    StructField('z',  DoubleType(), False),
    StructField('vx', DoubleType(), False),
    StructField('vy', DoubleType(), False),
    StructField('vz', DoubleType(), False),
    StructField('m',  DoubleType(), False)
])

"""default input schema - cluster data"""

clust_t = StructType([
    StructField('id', IntegerType(), False),
    StructField('x',  DoubleType(), False),
    StructField('y',  DoubleType(), False),
    StructField('z',  DoubleType(), False),
    StructField('vx', DoubleType(), False),
    StructField('vy', DoubleType(), False),
    StructField('vz', DoubleType(), False),
    StructField('m',  DoubleType(), False),
    StructField('t',  DoubleType(), False)
])

"""cluster data including, poistion, velocity, mass and time"""

F_cartesian = StructType([
    StructField('id',       IntegerType(), False),
    StructField('id_other', IntegerType(), False),
    StructField("Fx",       DoubleType(), False),
    StructField("Fy",       DoubleType(), False),
    StructField("Fz",       DoubleType(), False)
])

"""Schema including two ids and force, used in integrator_base"""

F_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("Fx", DoubleType(), False),
    StructField("Fy", DoubleType(), False),
    StructField("Fz", DoubleType(), False)
])

"""Schema including id and force, used in integrator_base"""

r_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("x",  DoubleType(), False),
    StructField("y",  DoubleType(), False),
    StructField("z",  DoubleType(), False)
])

"""Schema including the id and position, used in integrators"""

v_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("vx", DoubleType(), False),
    StructField("vy", DoubleType(), False),
    StructField("vz", DoubleType(), False)
])

"""Schema including the id and velocity, used in integrators"""

E_test_res = StructType([
    StructField('name',       StringType(), False),
    StructField("E",          DoubleType(), False),
    StructField("targetE",    DoubleType(), False),
    StructField("difference", DoubleType(), False),
])

"""Output schema for test.py"""

diag = StructType([
    StructField('t',   DoubleType(), False),
    StructField("E",   DoubleType(), False),
    StructField("dE",  DoubleType(), False),
])

"""Diagnostic output schema"""

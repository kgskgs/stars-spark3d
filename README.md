An N-body problem in physics aims to predict the movement of [stellar] particles that interact gravitationally. Such a simulation has an analytical solution only for two particles, and is computationally expensive as every two particles interact. 

This repo aims to implement some computation methods for conducting an n-body simulation with Apache Spark (pySpark more concretely). 

Detailed information and an API document can be found in nbody_simulations_spark.pdf, presented as a bachelors thesis for computer science at New Bulgarian University.

## Requirements 
* Python 3.6+
* Apache Spark 2+
* The python modules from ```requirements.txt```

## Usage

Currently, the most basic step method is implemented. It can be ran by submitting ```main.py``` as a Spark job, with the appropriate arguments to control the simulation (see ```python main.py -h``` for arguments).

```cluster.py``` contains functions that calculate basic statistics of the cluster, which are typically used to check the accuracy of the simulation.


## License
MIT
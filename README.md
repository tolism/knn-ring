# knn-ring

Parallel and Distributed Systems : 2nd assignment

## Installation and Execution 

To execute the versions make all inside the knnring file

./src/knnring_sequential for the sequential version 
mpirun -np <number of processes> ./src/knnring_mpi_syc for the sychronous version 
mpirun -np <number of processes> ./src/knnring_mpi_asyc for the asychronous version 

```
make all
/src/knnring_sequential
mpirun -np <number of processes> ./src/knnring_mpi_syc
mpirun -np <number of processes> ./src/knnring_mpi_asyc

```


These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

We will use the mpi library for the communication and the openblas for faster matrix calculations

```
#include <mpi.h>
#include "cblas"
```

## Authors

* **Apostolos Moustaklis**  
* **Savvas Christoforidis**  

# k-Nearest Neighbors Ring

Parallel and Distributed Systems : 2nd assignment

## Installation and Execution 

To execute the versions 
```
make all
/src/knnring_sequential 
mpirun -np <number of processes> ./src/knnring_mpi_syc
mpirun -np <number of processes> ./src/knnring_mpi_asyc

```
To change the number of poinnts , dimensions and k , modify the tester.c for the sequential and tester_mpi.c 
for the mpi versions.

### Prerequisites

We will use the mpi library for the communication and the openblas for faster matrix calculations

```
#include <mpi.h>
#include "cblas"
```

## Authors

* **Apostolos Moustaklis**  
* **Savvas Christoforidis**  

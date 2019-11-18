CC=mpicc
LOADER=gfortran
BLLIB = libblas.a	
CBLIB = cblas_LINUX.a

CFLAGS = -O3 -DADD_

all: knn

knn:
	$(CC) -c $(CFLAGS) cblas_knn.c
	$(LOADER) -o cblas_knn cblas_knn.o $(CBLIB) $(BLLIB)

clean:
	rm -f *.o cblas_knn

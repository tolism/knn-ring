CC = gcc
MPICC = mpicc
CFLAGS = -O3 -std=c99
SDIR=./src
LDIR=./lib
IDIR=./inc
EXE = $(SDIR)/knnring_sequential $(SDIR)/knnring_mpi
LIBS = $(LDIR)/knnring_sequential.a $(LDIR)/knnring_mpi.a
INCLUDES = $(IDIR)/knnring.h
MAIN = tester


all: $(EXE)
lib: $(LIBS)



$(SDIR)/knnring_sequential: $(SDIR)/$(MAIN).c $(LDIR)/knnring_sequential.a
	$(CC) -I$(IDIR) -L$(LDIR) -o $@ $^ -lopenblas -lm

$(SDIR)/knnring_mpi: $(SDIR)/$(MAIN)_mpi.c $(LDIR)/knnring_mpi.a
	$(MPICC) -I$(IDIR) -L$(LDIR) -o $@ $^ -lopenblas -lm





$(LDIR)/knnring_sequential.a: $(SDIR)/knnring_sequential.o
	ar rcs $@ $<

$(LDIR)/knnring_mpi.a: $(SDIR)/knnring_mpi.o
	ar rcs $@ $<





$(SDIR)/knnring_sequential.o:	$(SDIR)/knnring_sequential.c
	$(CC) -g -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm

$(SDIR)/knnring_mpi.o:	$(SDIR)/knnring_mpi.c
	$(MPICC) -g -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm




clean:
	rm -f $(SDIR)/*.o $(EXE) $(LIBS)

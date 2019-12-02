/*
* HELPER FILE: knnring_mpi.c
* Used for the Elearning Online Tester Only
* V1 and V2 files are knnring_mpi_syc and knnring_mpi_asyc

*/



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"
#include <mpi.h>


void swapElement(double ** one, double ** two);
int partition(double * arr, int * index, int l, int r);
void kthSmallest(double * arr, int * idx, int l, int r, int k);
void quicksort(double * array, int * idx, int first, int last);
knnresult updateResult(knnresult result, knnresult tempResult, int offset, int newOff);
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

knnresult distrAllkNN(double * X, int n, int d, int k) {

  int taskNum, taskid;
  MPI_Comm_size(MPI_COMM_WORLD, & taskNum);
  MPI_Comm_rank(MPI_COMM_WORLD, & taskid);

  int * idx = (int * ) malloc(n * k * sizeof(int));
  double * dist = (double * ) malloc(n * k * sizeof(double));

  knnresult result;
  knnresult tempResult;

  result.m = n;
  result.k = k;
  idx = result.nidx;
  dist = result.ndist;

  double * buffer = (double * ) malloc(n * d * sizeof(double));
  double * myElements = (double * ) malloc(n * d * sizeof(double));
  double * otherElements = (double * ) malloc(n * d * sizeof(double));
  
  myElements = X;
  int counter = 2;
  int offset, newOff;

  clock_t t1, t2, sum = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  t1 = clock();

  switch (taskid % 2) {
  case 0:
    MPI_Recv(otherElements, n * d, MPI_DOUBLE, (taskNum + taskid - 1) % taskNum, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    t2 = clock();
    result = kNN(myElements, myElements, n, n, d, k);
    sum += clock() - t2;
    MPI_Send(myElements, n * d, MPI_DOUBLE, (taskid + 1) % taskNum, 0, MPI_COMM_WORLD);
    t2 = clock();
    tempResult = kNN(otherElements, myElements, n, n, d, k);
    offset = (taskNum + taskid - 1) % taskNum;
    newOff = (taskNum + offset - 1) % taskNum;
    result = updateResult(result, tempResult, offset, newOff);
    sum += clock() - t2;

    while (counter < taskNum) {
      MPI_Recv(buffer, n * d, MPI_DOUBLE, (taskNum + taskid - 1) % taskNum, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(otherElements, n * d, MPI_DOUBLE, (taskid + 1) % taskNum, 0, MPI_COMM_WORLD);
      swapElement( & otherElements, & buffer);
      //otherElements = buffer;
      newOff = (taskNum + newOff - 1) % taskNum;
      t2 = clock();
      tempResult = kNN(otherElements, myElements, n, n, d, k);
      result = updateResult(result, tempResult, 0, newOff);
      sum += clock() - t2;
      counter++;
    }
    break;

  case 1:
    MPI_Send(myElements, n * d, MPI_DOUBLE, (taskid + 1) % taskNum, 0, MPI_COMM_WORLD);
    t2 = clock();
    result = kNN(myElements, myElements, n, n, d, k);
    sum += clock() - t2;
    MPI_Recv(otherElements, n * d, MPI_DOUBLE, taskid - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    t2 = clock();
    tempResult = kNN(otherElements, myElements, n, n, d, k);
    offset = (taskNum + taskid - 1) % taskNum;
    newOff = (taskNum + offset - 1) % taskNum;
    result = updateResult(result, tempResult, offset, newOff);
    sum += clock() - t2;

    while (counter < taskNum) {
      MPI_Send(otherElements, n * d, MPI_DOUBLE, (taskid + 1) % taskNum, 0, MPI_COMM_WORLD);
      MPI_Recv(otherElements, n * d, MPI_DOUBLE, taskid - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      t2 = clock();
      tempResult = kNN(otherElements, myElements, n, n, d, k);
      newOff = (taskNum + newOff - 1) % taskNum;
      result = updateResult(result, tempResult, 0, newOff);
      sum += clock() - t2;
      counter++;
    }
    break;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = clock() - t1;

  double computeTime = ((double) sum) / CLOCKS_PER_SEC;
  double avgComputeTime = 0;
  double timeTaken = ((double) t1) / CLOCKS_PER_SEC;
  double avgTimeTaken = 0;
  MPI_Reduce( & computeTime, & avgComputeTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce( & timeTaken, & avgTimeTaken, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (taskid == 0) {
    printf("Time taken for computing:  %lf\n", avgComputeTime / (taskNum));
    printf("Time taken for communication :  %lf\n", avgTimeTaken / (taskNum) - avgComputeTime / (taskNum));
  }

  double localMin = result.ndist[1];
  double localMax = result.ndist[0];
  for (int i = 0; i < n * k; i++) {
    if (result.ndist[i] > localMax) {
      localMax = result.ndist[i];
    }
    if (result.ndist[i] < localMin && result.ndist[i] != 0) {
      localMin = result.ndist[i];
    }
  }

  double globalMin;
  double globalMax;

  MPI_Allreduce( & localMin, & globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce( & localMax, & globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  //printf("AT process  %d MAX : %lf, MIN : %lf  \n " ,taskid , globalMax , globalMin );

  return result;
}

void swapElement(double ** one, double ** two) {
  double * temp = * one;
  * one = * two;
  * two = temp;
}

int partition(double * arr, int * index, int l, int r) {
  #	define SWAP(a, b) { tmp = arr[a]; arr[a] = arr[b]; arr[b] = tmp; }
  #	define SWAPINDEX(a, b) { tmpIdx = index[a]; index[a] = index[b]; index[b] = tmpIdx; }
  double tmp;
  int tmpIdx;

  double x = arr[r];
  int i = l;
  for (int j = l; j <= r - 1; j++) {
    if (arr[j] <= x) {
      SWAP(i, j);
      SWAPINDEX(i, j);

      i++;
    }
  }
  SWAP(i, r);
  SWAPINDEX(i, r);
  return i;
}

// This function returns k'th smallest
// element in arr[l..r] using QuickSort
// based method.  ASSUMPTION: ALL ELEMENTS
// IN ARR[] ARE DISTINCT
void kthSmallest(double * arr, int * idx, int l, int r, int k) {
  // If k is smaller than number of
  // elements in array
  if (k > 0 && k <= r - l + 1) {

    // Partition the array around last
    // element and get position of pivot
    // element in sorted array
    int index = partition(arr, idx, l, r);

    // If position is same as k
    if (index - l == k - 1)
      return;

    // If position is more, recur
    // for left subarray
    if (index - l > k - 1)
      return kthSmallest(arr, idx, l, index - 1, k);

    // Else recur for right subarray
    return kthSmallest(arr, idx, index + 1, r,
      k - index + l - 1);
  }

  // If k is more than number of
  // elements in array
  return;
}

void quicksort(double * array, int * idx, int first, int last) {
  int i, j, pivot;
  double temp;

  if (first < last) {
    pivot = first;
    i = first;
    j = last;

    while (i < j) {
      while (array[i] <= array[pivot] && i < last)
        i++;
      while (array[j] > array[pivot])
        j--;
      if (i < j) {
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;

        temp = idx[i];
        idx[i] = idx[j];
        idx[j] = temp;
      }
    }

    temp = array[pivot];
    array[pivot] = array[j];
    array[j] = temp;

    temp = idx[pivot];
    idx[pivot] = idx[j];
    idx[j] = temp;

    quicksort(array, idx, first, j - 1);
    quicksort(array, idx, j + 1, last);

  }
}

knnresult updateResult(knnresult result, knnresult tempResult, int offset, int newOff) {
  double * y = (double * ) malloc(result.m * result.k * sizeof(double));
  int * yidx = (int * ) malloc(result.m * result.k * sizeof(int));

  int p1, p2, p3;
  for (int i = 0; i < result.m; i++) {
    p1 = 0, p2 = 0, p3 = 0;
    while (p3 < result.k) {
      if ( * (result.ndist + i * result.k + p1) < * (tempResult.ndist + i * result.k + p2)) {
        *(y + i * result.k + p3) = * (result.ndist + i * result.k + p1);
        *(yidx + i * result.k + p3) = * (result.nidx + i * result.k + p1) + offset * result.m;
        p3++;
        p1++;
      } else {
        *(y + i * result.k + p3) = * (tempResult.ndist + i * result.k + p2);
        *(yidx + i * result.k + p3) = * (tempResult.nidx + i * result.k + p2) + newOff * result.m;
        p3++;
        p2++;
      }
    }
  }
  for (int i = 0; i < result.m; i++) {
    for (int j = 0; j < result.k; j++) {
      *(result.ndist + i * result.k + j) = * (y + i * result.k + j);
      *(result.nidx + i * result.k + j) = * (yidx + i * result.k + j);
    }
  }

  return result;
}

knnresult kNN(double * X, double * Y, int n, int m, int d, int k) {

  knnresult result;
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;

  double alpha = -2.0, beta = 0.0;
  int lda = d, ldb = d, ldc = m, i, j;
  double zerolim = 0.00000001;

  double * distance = (double * ) calloc((n * m), sizeof(double));
  double * sumX2 = (double * ) calloc(n, sizeof(double));
  double * sumY2 = (double * ) calloc(m, sizeof(double));
  double * distanceT = (double * ) malloc(m * n * sizeof(double));
  int * indeces = (int * ) malloc(m * n * sizeof(int));
  double * final = (double * ) malloc(m * k * sizeof(double));
  int * finalIdx = (int * ) malloc(m * k * sizeof(int));

  // set indeces
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      *(indeces + i * n + j) = j;
    }
  }

  // X, Y matrix multiplication using cblas: distance = -2 X * Y.'
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, alpha, X, lda, Y, ldb, beta, distance, ldc);

  // sum(X.^2,2) calculation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      sumX2[i] += ( * (X + i * d + j)) * ( * (X + i * d + j));
    }
  }
  // sum(Y.^2,2).' calculation
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < d; j++) {
      sumY2[i] += ( * (Y + i * d + j)) * ( * (Y + i * d + j));
    }
  }

  // distance addition formula
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      *(distance + i * m + j) += sumX2[i] + sumY2[j];
      if ( * (distance + i * m + j) < zerolim) {
        *(distance + i * m + j) = 0;
      } else {
        *(distance + i * m + j) = sqrt( * (distance + i * m + j));
      }
    }
  }
  free(sumX2);
  free(sumY2);

  // calculate transpose matrix of distance
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      *(distanceT + j * n + i) = * (distance + i * m + j);;
    }
  }
  free(distance);

  // moving kthSmallest to k first columns
  for (int i = 0; i < m; i++) {
    kthSmallest(distanceT, indeces, i * n, (i + 1) * n - 1, k);
  }

  // array cut: m*n -> m*k
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      *(final + i * k + j) = * (distanceT + i * n + j);
      *(finalIdx + i * k + j) = * (indeces + i * n + j);
    }
  }

  // sort each row
  for (int i = 0; i < m; i++) {
    quicksort(final, finalIdx, i * k, (i + 1) * k - 1);
  }
  free(distanceT);
  free(indeces);

  result.ndist = final;
  result.nidx = finalIdx;

  return result;
}

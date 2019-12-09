/*
* FILE: knnring_mpi_asyc.c
* THMMY, 7th semester, Parallel and Distributed Systems: 2nd assignment
* MPI implementation of knnring
* Authors:
*   Moustaklis Apostolos, 9127, amoustakl@ece.auth.gr
*   Christoforidis Savvas, 9147, schristofo@ece.auth.gr
* Compile command with :
*   make all
* Run command example:
*   ./knnring_mpi_asyc
* It will find the k-Nearest Neighbours
* of the given corpus and query set
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"
#include <mpi.h>


//Function declaration
void swapElement(double **one, double  **two);
int partition(double *arr , int *index, int l, int r);
void kthSmallest(double *arr,int *idx , int l, int r, int k);
void quicksort(double *array, int *idx, int first, int last);
knnresult updateResult(knnresult result,knnresult tempResult,int offset,int newOff);
knnresult kNN(double * X , double * Y , int n , int m , int d , int k);

// knn Function for the distributed - ring calculation
knnresult distrAllkNN(double * X , int n , int d , int k ) {

  int taskNum , taskid ;
  MPI_Comm_size(MPI_COMM_WORLD,&taskNum);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Request request[3];
  MPI_Status status;

  //Allocating memory
  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));


  knnresult result ;
  knnresult tempResult  ;

  result.m=n;
  result.k=k;



  myElements = X;
  /*
     Initialize counter for counting how many data
     have been calculated in each process in order
     to know when to finish the KNN algorithm.
    */
  int counter= 2;

  //Variables used to find the offset of the indeces
  int newOff , offset;

  //Variable used to calculate the computation and communication time
  clock_t t1;

  MPI_Barrier(MPI_COMM_WORLD);

  t1 = clock();

  /*Move data into circle
  Even id's receive
  Odd id's  send
  */

  switch(taskid%2){
    case 0:
    MPI_Isend(myElements , n*d , MPI_DOUBLE, (taskid + 1)%taskNum , 0 , MPI_COMM_WORLD , &request[0] );
    MPI_Irecv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
    result = kNN(myElements,myElements,n,n,d,k);
    //Finding the offset of the indeces
    offset = (taskNum+taskid-1)%taskNum;
    newOff = (taskNum + offset-1)%taskNum;
    MPI_Wait(&request[1],&status);

    while(counter<taskNum){
      MPI_Isend(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%taskNum , 0 , MPI_COMM_WORLD , &request[2] );
      MPI_Irecv(buffer , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
        tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

        if(counter == 2 ){
        result = updateResult( result, tempResult, offset, newOff);
        }
        else{
          //Updating the offset
          newOff = (taskNum + newOff-1)%taskNum;
          result = updateResult( result, tempResult, 0, newOff);
        }
      MPI_Wait(&request[1],&status);
      MPI_Wait(&request[2],&status);
      swapElement(&otherElements,&buffer);
        counter++;
      }

      tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
      if(taskNum!=2){
      newOff = (taskNum + newOff-1)%taskNum;
      offset = 0 ;
      }
      result = updateResult( result, tempResult, offset, newOff);
      break;


      case 1:
      MPI_Isend(myElements , n*d , MPI_DOUBLE, (taskid + 1)%taskNum , 0 , MPI_COMM_WORLD , &request[0] );
      MPI_Irecv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
      result = kNN(myElements,myElements,n,n,d,k);
      //Finding the offset of the indeces
      offset = (taskNum+taskid-1)%taskNum;
      newOff = (taskNum + offset-1)%taskNum;
      MPI_Wait(&request[1],&status);
      while(counter<taskNum){
        MPI_Isend(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%taskNum , 0 , MPI_COMM_WORLD , &request[2] );
        MPI_Irecv(buffer , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
          tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

          if(counter == 2 ){
          result = updateResult( result, tempResult, offset, newOff);
          }
          else{
            //Updating the offset
            newOff = (taskNum + newOff-1)%taskNum;
            result = updateResult( result, tempResult, 0, newOff);
          }
        MPI_Wait(&request[1],&status);
        MPI_Wait(&request[2],&status);
        swapElement(&otherElements,&buffer);
          counter++;
        }
        tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
        if(taskNum!=2){
        newOff = (taskNum + newOff-1)%taskNum;
        offset = 0 ;
        }
        result = updateResult( result, tempResult, offset , newOff);
        break;

}
MPI_Barrier(MPI_COMM_WORLD);

t1 = clock() - t1;
double timeTaken = ((double) t1) / CLOCKS_PER_SEC;
double avgTimeTaken = 0;
MPI_Reduce( & timeTaken, & avgTimeTaken, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if (taskid == 0) {
  printf("Time taken :  %lf\n", timeTaken / (taskNum));

}

double localMin=result.ndist[1];
double localMax=result.ndist[0];
for(int i=0; i <n*k; i++){
  if(result.ndist[i]>localMax){
    localMax = result.ndist[i];
  }
  if(result.ndist[i]<localMin && result.ndist[i]!=0){
    localMin = result.ndist[i];
  }
}


double globalMin;
double globalMax;

MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
if(!taskid)
printf("AT process  %d MAX : %lf, MIN : %lf  \n " ,taskid , globalMax , globalMin );



  return result;
}


//Swap function
void swapElement(double ** one, double ** two) {
  double * temp = * one;
  * one = * two;
  * two = temp;
}


//Partition used in quick select
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


//Quick-sort algorithm
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


// Helper function to update the results
knnresult updateResult(knnresult result, knnresult tempResult, int offset, int newOff) {
  double * y = (double * ) malloc(result.m * result.k * sizeof(double));
  int * yidx = (int * ) malloc(result.m * result.k * sizeof(int));

  //We use 3 counters to iterate the 2 sets and keep only the k smallest one
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

//kNN Function
knnresult kNN(double * X, double * Y, int n, int m, int d, int k) {

  knnresult result;
  result.k = k;
  result.m = m;


  double alpha = -2.0, beta = 0.0;
  int lda = d, ldb = d, ldc = m;
  double zerolim   = 0.00000001;

  double * distance = (double * ) calloc((n * m), sizeof(double));
  double * sumX2 = (double * ) calloc(n, sizeof(double));
  double * sumY2 = (double * ) calloc(m, sizeof(double));
  double * distanceT = (double * ) malloc(m * n * sizeof(double));
  int * indeces = (int * ) malloc(m * n * sizeof(int));
  double * final = (double * ) malloc(m * k * sizeof(double));
  int * finalIdx = (int * ) malloc(m * k * sizeof(int));

  // set indices
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      *(indeces + i * n + j) = j;
    }
  }

  // X,Y matrix multiplication using cblas: distance = -2 X * Y.'
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

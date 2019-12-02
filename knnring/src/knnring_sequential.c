/*
* FILE: knnring_sequential.c
* THMMY, 7th semester, Parallel and Distributed Systems: 2nd assignment
* Sequential implementation of knnring
* Authors:
*   Moustaklis Apostolos, 9127, amoustakl@ece.auth.gr
*   Christoforidis Savvas, 9147, schristofo@ece.auth.gr
* Compile command with :
*   make all
* Run command example:
*   ./src/knnring_sequential
* It will find the k-Nearest Neighbours
* of the given corpus and query set
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"

int partition(double *arr , int *index, int l, int r);
void kthSmallest(double *arr,int *idx , int l, int r, int k);
void quicksort(double *array, int *idx, int first, int last);

knnresult kNN(double * X , double * Y , int n , int m , int d , int k);



knnresult kNN(double * X , double * Y , int n , int m , int d , int k) {


  knnresult result;
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;

  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;
  double zerolim = 0.00000001;

  double * distance = (double *) calloc((n*m),sizeof(double));
  double * sumX2 = (double *) calloc(n,sizeof(double));
  double * sumY2 = (double *) calloc(m,sizeof(double));
  double * distanceT = (double *) malloc(m*n*sizeof(double));
  int * indeces = (int*) malloc(m * n  *sizeof(int));
  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));

  // set indices
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  // X,Y matrix multiplication using cblas: distance = -2 X * Y.'
  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);

  // sum(X.^2,2) calculation
  for(int i=0; i<n; i++){
    for(int j=0; j<d; j++){
      sumX2[i] += (*(X+i*d+j)) * (*(X+i*d+j));
    }
  }
  // sum(Y.^2,2).' calculation
  for(int i=0; i<m; i++){
    for(int j=0; j<d; j++){
      sumY2[i] += (*(Y+i*d+j)) * (*(Y+i*d+j));
    }
  }

  // distance addition formula
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(distance + i*m + j) += sumX2[i] + sumY2[j];
      if(*(distance + i*m + j) < zerolim){
        *(distance + i*m + j) = 0;
      }
      else{
        *(distance + i*m + j) = sqrt( *(distance + i*m + j) );
      }
    }
  }
  free(sumX2);
  free(sumY2);

  // calculate transpose matrix of distance
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(distanceT + j*n + i ) = *(distance + i*m + j ); ;
    }
  }
//  free(distance);

  // moving kthSmallest to k first columns
  for(int i=0; i < m; i++){
    kthSmallest(distanceT , indeces , i*n , (i+1)*n-1, k);
  }

  // array cut: m*n -> m*k
  for(int i = 0; i<m; i++){
    for(int j = 0; j<k; j++){
      *(final+i*k+j) = *(distanceT+i*n+j);
      *(finalIdx+i*k+j) = *(indeces+i*n+j);
    }
  }

  // sort each row
  for(int i = 0 ; i<m; i++){
      quicksort(final , finalIdx , i*k , (i+1)*k-1);
  }

  free(distanceT);
  free(indeces);

  result.ndist = final;
  result.nidx = finalIdx;

  return result;
}



int partition(double *arr , int *index, int l, int r)
{
  #	define SWAP(a, b) { tmp = arr[a]; arr[a] = arr[b]; arr[b] = tmp; }
  #	define SWAPINDEX(a, b) { tmpIdx = index[a]; index[a] = index[b]; index[b] = tmpIdx; }
    double tmp;
    int tmpIdx;

    double x = arr[r];
    int  i = l;
    for (int j = l; j <= r - 1; j++) {
        if (arr[j] <= x) {
          SWAP(i, j);
          SWAPINDEX(i,j);

            i++;
        }
    }
    SWAP(i, r);
    SWAPINDEX(i,r);
    return i;
}

// This function returns k'th smallest
// element in arr[l..r] using QuickSort
// based method.  ASSUMPTION: ALL ELEMENTS
// IN ARR[] ARE DISTINCT
void kthSmallest(double *arr,int *idx , int l, int r, int k)
{
    // If k is smaller than number of
    // elements in array
    if (k > 0 && k <= r - l + 1) {

        // Partition the array around last
        // element and get position of pivot
        // element in sorted array
        int index = partition(arr,idx ,  l, r);

        // If position is same as k
        if (index - l == k - 1)
            return ;

        // If position is more, recur
        // for left subarray
        if (index - l > k - 1)
            return kthSmallest(arr, idx , l, index - 1, k);

        // Else recur for right subarray
        return kthSmallest(arr, idx , index + 1, r,
                            k - index + l - 1);
    }

    // If k is more than number of
    // elements in array
    return;
}

void quicksort(double *array, int *idx, int first, int last){
   int i, j, pivot;
   double  temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(array[i]<=array[pivot]&&i<last)
            i++;
         while(array[j]>array[pivot])
            j--;
         if(i<j){
            temp=array[i];
            array[i]=array[j];
            array[j]=temp;

            temp=idx[i];
            idx[i]=idx[j];
            idx[j]=temp;
         }
      }

      temp=array[pivot];
      array[pivot]=array[j];
      array[j]=temp;

      temp=idx[pivot];
      idx[pivot]=idx[j];
      idx[j]=temp;

      quicksort(array,idx,first,j-1);
      quicksort(array,idx,j+1,last);

   }
}

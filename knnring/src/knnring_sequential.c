#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"


typedef struct distIdx {
  double distance;
  int index;
} distIdx;

distIdx qselect(double *tArray,int *index, int len, int k) {
	#	define SWAP(a, b) { tmp = tArray[a]; tArray[a] = tArray[b]; tArray[b] = tmp; }
  #	define SWAPINDEX(a, b) { tmp = index[a]; index[a] = index[b]; index[b] = tmp; }
	int i, st;
	double tmp;
  distIdx c;
	// double * tArray = (double * ) malloc(len * sizeof(double));
	// for(int i=0; i<len; i++){
	// 	tArray[i] = v[i];
	// }
	for (st = i = 0; i < len - 1; i++) {
		if (tArray[i] > tArray[len-1]) continue;
		SWAP(i, st);
    SWAPINDEX(i,st);
		st++;
	}
	SWAP(len-1, st);
  SWAPINDEX(len-1,st);
  if(k < st){
    c = qselect(tArray, index,st, k);
  }
  else if(k > st){
    c = qselect(tArray + st, index + st, len - st, k - st);
  }
  if (k == st){
    c.distance = tArray[st];
    c.index = index[st];
    return c;
  }
  return c;
	//return k == st	? tArray[st] : st > k	? qselect(tArray, st, k) : qselect(tArray + st, len - st, k - st);
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




knnresult kNN(double * X , double * Y , int n , int m , int d , int k) {

  knnresult result;
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;

  distIdx p;

  //X: n * d
  //Y: m * d
  double * distance;
  int *indeces;
  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;
  int counter = 0;

  distance = (double *) malloc((n*m)*sizeof(double));

  indeces= (int*)malloc(m * n  *sizeof(int));

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);


  double * xRow = (double *) calloc(n,sizeof(double));
  double * yRow = (double *) calloc(m,sizeof(double));

  for(int i=0; i<n; i++){
    for(int j=0; j<d; j++){
      xRow[i] += (*(X+i*d+j)) * (*(X+i*d+j));
    }
  }
  for(int i=0; i<m; i++){
    for(int j=0; j<d; j++){
      yRow[i] += (*(Y+i*d+j)) * (*(Y+i*d+j));
    }
  }

  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(distance + i*m + j) += xRow[i] + yRow[j];
      if(*(distance + i*m + j) < 0.00000001){
        *(distance + i*m + j) = 0;
      }
      else{
      *(distance + i*m + j) = sqrt( *(distance + i*m + j) );
    }
  }
  }
  free(xRow);
  free(yRow);

  // calculate transpose matrix
  double * transD = (double *) malloc(m*n*sizeof(double));
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i ) = *(distance + i*m + j );
    }
  }

  // distance = transD then delete transD
  for(int i=0; i<n*m; i++) {
    *(distance+i) = *(transD+i);
  }
  free(transD);
  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));
  double * temp = (double *) malloc(n * sizeof(double));
  int * tempIdx = (int *) malloc (n * sizeof(int));
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      *(temp+j) = *(distance+i*n+j);
      *(tempIdx+j)= *(indeces+i*n+j);
    }
    p = qselect(temp,tempIdx,n,k);
    quicksort(temp, tempIdx,0,k);
    for(int j=0; j<k; j++){
      *(final+i*k+j) = temp[j];
      *(finalIdx+i*k+j) = tempIdx[j];
    }
  }



  result.ndist = final;
  result.nidx = finalIdx;

  return result;
}

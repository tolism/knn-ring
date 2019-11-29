#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"
#include <mpi.h>


typedef struct distIdx {
  double distance;
  int index;
} distIdx;

void swapElement(double **one, double  **two){
	double  *temp = *one;
	*one = *two;
	*two = temp;
}
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
  int taskid, numtasks;
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

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

    // EGINE ALLAGH EDW GIA INDEX APO QUERY
    //  *(indeces+i*n+j)=j;
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



knnresult distrAllkNN(double * X , int n , int d , int k ) {

  int numtasks , taskid ;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

  int *idx =(int *)malloc(n*k*sizeof(int));
  double * dist = (double *) malloc(n * k * sizeof(double));

  knnresult result ;
  knnresult tempResult  ;
//  printf("2 \n ");
  result.m=n;
  result.k=k;
  // EAN THELEI TA INDEX APO TO KATHENA ME THN METHODO ALVANIAS
  // THA POLLAPLASIASW TO INDEX * TASKID GIA NA VGEI SE KLIMAKA POU THELEI
  idx = result.nidx;
  dist = result.ndist;

  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));
  double *y = (double *)malloc(n*k*sizeof(double));
  int *yidx = (int *)malloc(n*k*sizeof(int));
  myElements = X;
  int counter= 2;
  // NA ALLAKSOUME KAI NA TO VALOYME IDIO SE OLA EKTOS APO TO MHDEN POU THELEI POUTANIA
  // GIA NA EINAI KUKLIKO TO RING
  int p1, p2, p3;
  if(taskid%2){
  MPI_Send(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
  result = kNN(myElements,myElements,n,n,d,k);
  MPI_Recv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  tempResult = kNN(myElements, otherElements , n , n , d ,k);
  int offset = (numtasks+taskid-1)%numtasks;
  int newOff = (numtasks+offset-1)%numtasks;

  for(int i=0; i<n; i++){
    p1=0, p2=0, p3=0;
    while (p3<k) {
        if (*(result.ndist + i*k + p1) < *(tempResult.ndist + i*k+ p2)){
          *(y+i*k+p3) = *(result.ndist+ i*k+p1);
          *(yidx+i*k+p3) = *(result.nidx+i*k+p1) + offset*n;
          p3++;
          p1++;
        }
        else{
          *(y+i*k+p3) = *(tempResult.ndist+i*k+p2);
          *(yidx+i*k+p3) = *(tempResult.nidx+i*k+p2) + newOff*n  ;
          p3++;
          p2++;
        }
    }
  }
  for(int i=0; i<n; i++){
    for(int j = 0 ; j <k ; j++){
      *(result.ndist+i*k+j) = *(y+i*k+j);
      *(result.nidx+i*k+j)= *(yidx+i*k+j);
    }
  }
    while(counter<numtasks){
      MPI_Send(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
      MPI_Recv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      tempResult = kNN(myElements, otherElements , n , n , d ,k );
      newOff = (numtasks + newOff-1)%numtasks;
      for(int i=0; i<n; i++){
        p1=0, p2=0, p3=0;
        while (p3<k) {
            if (*(result.ndist + i*k + p1) < *(tempResult.ndist + i*k+ p2)){
              *(y+i*k+p3) = *(result.ndist+ i*k+p1);
              *(yidx+i*k+p3) = *(result.nidx+i*k+p1);
              p3++;
              p1++;
            }
            else{
              *(y+i*k+p3) = *(tempResult.ndist+i*k+p2);
              *(yidx+i*k+p3) = *(tempResult.nidx+i*k+p2) + newOff*n;
              p3++;
              p2++;
            }
        }
      }
      for(int i=0; i<n; i++){
        for(int j = 0 ; j <k ; j++){
        *(result.ndist+i*k+j) = *(y+i*k+j);
        *(result.nidx+i*k+j)= *(yidx+i*k+j);
      }
    }
      counter++;
    }
  }
  else{
      MPI_Recv(otherElements , n*d , MPI_DOUBLE, (numtasks+taskid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result = kNN(myElements,myElements,n,n,d,k);
      MPI_Send(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
      tempResult = kNN(myElements, otherElements , n , n , d ,k);
      int offset = (numtasks+taskid-1)%numtasks;
      int newOff = (numtasks+offset-1)%numtasks;
      int temp = 0;
      for(int i=0; i<n; i++){
        p1=0, p2=0, p3=0;
        while (p3<k) {
            if (*(result.ndist + i*k + p1) <= *(tempResult.ndist + i*k+ p2)){
              *(y+i*k+p3) = *(result.ndist+ i*k+p1);
              *(yidx+i*k+p3) = *(result.nidx+i*k+p1) + offset*n ;
              p3++;
              p1++;
            }
            else{
              *(y+i*k+p3) = *(tempResult.ndist+i*k+p2);
              *(yidx+i*k+p3) = *(tempResult.nidx+i*k+p2) + newOff*n;
              p3++;
              p2++;
            }
        }
      }
      for(int i=0; i<n; i++){
        for(int j = 0 ; j <k ; j++){
          *(result.ndist+i*k+j) = *(y+i*k+j);
          *(result.nidx+i*k+j)= *(yidx+i*k+j);
        }
      }

    while(counter<numtasks){
      MPI_Recv(buffer , n*d , MPI_DOUBLE, (numtasks+taskid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
      swapElement(&otherElements, &buffer);
      //otherElements = buffer;

      newOff = (numtasks + newOff-1)%numtasks;
      tempResult = kNN(myElements, otherElements , n ,n , d ,k );

      for(int i=0; i<n; i++){
        p1=0, p2=0, p3=0;
        while (p3<k) {
            if (*(result.ndist + i*k + p1) <= *(tempResult.ndist + i*k+ p2)) {
              *(y+i*k+p3) = *(result.ndist+ i*k+p1);
              *(yidx+i*k+p3) = *(result.nidx+i*k+p1);
              p3++;
              p1++;
            }
            else{
              *(y+i*k+p3) = *(tempResult.ndist+i*k+p2);
              *(yidx+i*k+p3) = *(tempResult.nidx+i*k+p2)+newOff*n;
              p3++;
              p2++;
            }
      }
      for(int i=0; i<n; i++){
        for(int j = 0 ; j <k ; j++){
          *(result.ndist+i*k+j) = *(y+i*k+j);
          *(result.nidx+i*k+j)= *(yidx+i*k+j);
        }
      }
      counter++;
    }
  }
}
  return result;
}

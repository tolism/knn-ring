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


knnresult updateResult(knnresult result,knnresult tempResult,int offset,int newOff){
  double *y = (double *)malloc(result.m*result.k*sizeof(double));
  int *yidx = (int *)malloc(result.m*result.k*sizeof(int));
  int p1 , p2 , p3;
  for(int i=0; i<result.m; i++){
    p1=0, p2=0, p3=0;
    while (p3<result.k) {
        if (*(result.ndist + i*result.k + p1) < *(tempResult.ndist + i*result.k+ p2)){
          *(y+i*result.k+p3) = *(result.ndist+ i*result.k+p1);
          *(yidx+i*result.k+p3) = *(result.nidx+i*result.k+p1) + offset*result.m;
          p3++;
          p1++;
        }
        else{
          *(y+i*result.k+p3) = *(tempResult.ndist+i*result.k+p2);
          *(yidx+i*result.k+p3) = *(tempResult.nidx+i*result.k+p2) + newOff*result.m  ;
          p3++;
          p2++;
        }
    }
  }
  for(int i=0; i<result.m; i++){
    for(int j = 0 ; j <result.k ; j++){
      *(result.ndist+i*result.k+j) = *(y+i*result.k+j);
      *(result.nidx+i*result.k+j)= *(yidx+i*result.k+j);
    }
  }
  
  return result;
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

  MPI_Request request[3];
  MPI_Status status;

  int *idx =(int *)malloc(n*k*sizeof(int));
  double * dist = (double *) malloc(n * k * sizeof(double));


  knnresult result ;
  knnresult tempResult  ;

  result.m=n;
  result.k=k;
  idx = result.nidx;
  dist = result.ndist;

  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));

  myElements = X;
  int counter= 2;
  int p1, p2, p3;
  int newOff , offset;




  if(taskid%2){
  MPI_Isend(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[0] );
  MPI_Irecv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
  result = kNN(myElements,myElements,n,n,d,k);
  offset = (numtasks+taskid-1)%numtasks;
  newOff = (numtasks + offset-1)%numtasks;
  MPI_Wait(&request[1],&status);


  while(counter<numtasks){
    MPI_Isend(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[2] );
    MPI_Irecv(buffer , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
      tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

      if(counter == 2 ){
      result = updateResult( result, tempResult, offset, newOff);
      }
      else{
        newOff = (numtasks + newOff-1)%numtasks;
        result = updateResult( result, tempResult, 0, newOff);
      }
    MPI_Wait(&request[1],&status);
    MPI_Wait(&request[2],&status);
    swapElement(&otherElements,&buffer);
      counter++;
    }
    tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
    newOff = (numtasks + newOff-1)%numtasks;
    result = updateResult( result, tempResult, 0, newOff);
  }
  else{
    MPI_Isend(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[0] );
    MPI_Irecv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
    result = kNN(myElements,myElements,n,n,d,k);
    offset = (numtasks+taskid-1)%numtasks;
    newOff = (numtasks + offset-1)%numtasks;
    MPI_Wait(&request[1],&status);


    while(counter<numtasks){
      MPI_Isend(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[2] );
      MPI_Irecv(buffer , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
        tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

        if(counter == 2 ){
        result = updateResult( result, tempResult, offset, newOff);
        }
        else{
          newOff = (numtasks + newOff-1)%numtasks;
          result = updateResult( result, tempResult, 0, newOff);
        }
      MPI_Wait(&request[1],&status);
      MPI_Wait(&request[2],&status);
      swapElement(&otherElements,&buffer);
        counter++;
      }
      tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
      newOff = (numtasks + newOff-1)%numtasks;
      result = updateResult( result, tempResult, 0, newOff);
}
MPI_Barrier(MPI_COMM_WORLD);

  return result;
}
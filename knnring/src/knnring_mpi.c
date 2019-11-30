#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"
#include <mpi.h>



void swapElement(double **one, double  **two);
void qselect(double *tArray,int *index, int len, int k);
void quicksort(double *array, int *idx, int first, int last);
knnresult updateResult(knnresult result,knnresult tempResult,int offset,int newOff);
knnresult kNN(double * X , double * Y , int n , int m , int d , int k);


knnresult distrAllkNN(double * X , int n , int d , int k ) {

  int numtasks , taskid ;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

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
  double *y = (double *)malloc(n*k*sizeof(double));
  int *yidx = (int *)malloc(n*k*sizeof(int));
  myElements = X;
  int counter= 2;
  int p1, p2, p3;
  int offset , newOff ;


  switch(taskid%2){
    case 0:
    MPI_Recv(otherElements , n*d , MPI_DOUBLE, (numtasks+taskid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    result = kNN(myElements,myElements,n,n,d,k);
    MPI_Send(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
    tempResult = kNN(otherElements , myElements , n , n , d ,k);
    offset = (numtasks+taskid-1)%numtasks;
    newOff = (numtasks+offset-1)%numtasks;
    result = updateResult( result, tempResult, offset, newOff);

  while(counter<numtasks){
    MPI_Recv(buffer , n*d , MPI_DOUBLE, (numtasks+taskid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
    swapElement(&otherElements, &buffer);
    //otherElements = buffer;
    newOff = (numtasks + newOff-1)%numtasks;
    tempResult = kNN( otherElements , myElements,  n ,n , d ,k );
    result = updateResult( result, tempResult, 0, newOff);
    counter++;
    }
    break;

    case 1:
    MPI_Send(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
    result = kNN(myElements,myElements,n,n,d,k);
    MPI_Recv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tempResult = kNN(otherElements , myElements,  n , n , d ,k);
    offset = (numtasks+taskid-1)%numtasks;
    newOff = (numtasks+offset-1)%numtasks;
    result = updateResult( result, tempResult, offset, newOff);

    while(counter<numtasks){
        MPI_Send(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
        MPI_Recv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
        newOff = (numtasks + newOff-1)%numtasks;
        result = updateResult( result, tempResult, 0, newOff);
        counter++;
      }
      break;
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

printf("AT process  %d MAX : %lf, MIN : %lf  \n " ,taskid , globalMax , globalMin );
  return result;

}

void swapElement(double **one, double  **two){
	double  *temp = *one;
	*one = *two;
	*two = temp;
}


void qselect(double *tArray,int *index, int len, int k) {
	#	define SWAP(a, b) { tmp = tArray[a]; tArray[a] = tArray[b]; tArray[b] = tmp; }
  #	define SWAPINDEX(a, b) { tmp = index[a]; index[a] = index[b]; index[b] = tmp; }
	int i, st;
	double tmp;

	for (st = i = 0; i < len - 1; i++) {
		if (tArray[i] > tArray[len-1]) continue;
		SWAP(i, st);
    SWAPINDEX(i,st);
		st++;
	}
	SWAP(len-1, st);
  SWAPINDEX(len-1,st);
  if(k < st){
    qselect(tArray, index,st, k);
  }
  else if(k > st){
    qselect(tArray + st, index + st, len - st, k - st);
  }
  if (k == st){
    return ;
  }
  return ;
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

  if(y==NULL){
    printf("Y EXEI THEMA");

  }
  if(yidx==NULL){
    printf("YIDX EXEI THEMA");

  }
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


  double * distance;
  int *indeces;
  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;
  int counter = 0;

  distance = (double *) malloc((n*m)*sizeof(double));
  double * xRow = (double *) calloc(n,sizeof(double));
  double * yRow = (double *) calloc(m,sizeof(double));
  double * transD = (double *)malloc(m*n*sizeof(double));
  indeces= (int*)malloc(m * n  *sizeof(int));
  if(distance == NULL){
    printf("distance exei thema");

  }

  if(indeces ==NULL ){
    printf("indeces exei thema ");

  }

  if(transD==NULL){
    printf("transd exei thema \n");

  }
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);


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
  //free(xRow);
  //free(yRow);
  // calculate transpose matrix

  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i )   = *(distance + i*m + j );
    }
  }
  //free(distance);

  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));
  double * temp = (double *) malloc(n * sizeof(double));
  int * tempIdx = (int *) malloc (n * sizeof(int));
  if(final==NULL){
    printf(" FINAL  EXEI THEMA");

  }
  if(finalIdx==NULL){
    printf(" finalidx  EXEI THEMA");

  }
  if(temp==NULL){
    printf(" temp EXEI THEMA");

  }
  if(tempIdx==NULL){
    printf(" tempidx  EXEI THEMA");

  }
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      *(temp+j) = *(transD+i*n+j);
      *(tempIdx+j)= *(indeces+i*n+j);
    }
    qselect(temp,tempIdx,n,k);
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

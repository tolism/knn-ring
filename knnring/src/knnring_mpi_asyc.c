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



void swapElement(double **one, double  **two);
int partition(double *arr , int *index, int l, int r);
void kthSmallest(double *arr,int *idx , int l, int r, int k);
void quicksort(double *array, int *idx, int first, int last);
knnresult updateResult(knnresult result,knnresult tempResult,int offset,int newOff);
knnresult kNN(double * X , double * Y , int n , int m , int d , int k);




knnresult distrAllkNN(double * X , int n , int d , int k ) {

  int numtasks , taskid ;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Request request[3];
  MPI_Status status;


  int *idx =(int *)malloc(n*k*sizeof(int));
  double * dist = (double *) malloc(n * k * sizeof(double));
  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));
   if(idx==NULL){
     printf("IDX THEMA ");

   }
   if(dist==NULL){
     printf("DIST THEMA");

   }
   if(buffer==NULL){
     printf("BUFFER THEMA");

   }
   if(myElements==NULL){
     printf("MyElements THEM");

   }
   if(otherElements==NULL){
     printf("OTHER ELEMENTS THEMA");

   }


  knnresult result ;
  knnresult tempResult  ;

  result.m=n;
  result.k=k;
  idx = result.nidx;
  dist = result.ndist;


  myElements = X;

  int counter= 2;
  int newOff , offset;

  switch(taskid%2){
    case 0:
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
      if(numtasks!=2){
      newOff = (numtasks + newOff-1)%numtasks;
      offset = 0 ;
      }
      result = updateResult( result, tempResult, offset, newOff);
      break;


      case 1:
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
        if(numtasks!=2){
        newOff = (numtasks + newOff-1)%numtasks;
        offset = 0 ;
        }
        result = updateResult( result, tempResult, offset , newOff);
        break;

}
MPI_Barrier(MPI_COMM_WORLD);

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

  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;
  int counter = 0;
  double limit = 0.00000001;

  double * distance = (double *) malloc((n*m)*sizeof(double));
  double * xRow = (double *) calloc(n,sizeof(double));
  double * yRow = (double *) calloc(m,sizeof(double));
  double * transD = (double *) malloc(m*n*sizeof(double));
  int * indeces = (int*) malloc(m * n  *sizeof(int));
  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  if(transD==NULL){
    printf("transd exei thema \n");
  }
  if(distance == NULL){
    printf("distance exei thema");
  }
  if(indeces ==NULL ){
    printf("indeces exei thema ");
  }
  if(final==NULL){
    printf(" FINAL  EXEI THEMA");
  }
  if(finalIdx==NULL){
    printf(" finalidx  EXEI THEMA");
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
      if(*(distance + i*m + j) < limit){
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
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i ) = *(distance + i*m + j ); ;
    }
  }
//  free(distance);

for(int i=0; i < m; i++){
  kthSmallest(transD , indeces , i*n , (i+1)*n-1, k);
}

for(int i = 0; i<m; i++){
  for(int j = 0; j<k; j++){

    *(final+i*k+j) = *(transD+i*n+j);
    *(finalIdx+i*k+j) = *(indeces+i*n+j);

  }
}
for(int i = 0 ; i<m; i++){
    quicksort(final , finalIdx , i*k , (i+1)*k-1);
}
 free(transD);
  free(indeces);

  result.ndist = final;
  result.nidx = finalIdx;

  return result;
}

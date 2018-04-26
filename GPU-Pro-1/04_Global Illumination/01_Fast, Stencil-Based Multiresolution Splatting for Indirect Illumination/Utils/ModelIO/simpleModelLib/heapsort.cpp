#include <stdio.h>
#include <stdlib.h>
#include "mesh.h"

static  Node * a[65535];
static  int N;

int NodeCompare(Node *a, Node *b){

  if( a->v < b->v ) return -1;
  if( a->v > b->v ) return  1;
  return 0;
}

void construct(Node * b[],int M){
  for (N=1;N<=M;N++) a[N]=b[N];
  if( M== 0 ) N = 0;
}

void upheap( int k ){
  Node* v;
  v = a[k];a[0] = NIL;
  while( NodeCompare(a[k/2],v) >= 0 && k > 1 )
    {
      a[k] = a[k/2];k=k/2;
    }
      a[k]=v;
}


void insert(Node * v)
{ a[++N]=v;
  upheap(N);
}


void downheap(int k){

 int j;
 Node * v;
 v = a[k];
 while( k <= N/2 ){
   j= k+k;
   if(j<N && NodeCompare(a[j],a[j+1])>0) j++;
   if(NodeCompare(v,a[j])<=0) break;
   a[k]= a[j];k=j;
 }
 
 a[k] = v;
 
}

Node * Remove(){
  Node * v = a[1];
  a[1] =  a[N--];
  downheap(1); 
  return v;
}

Node * replace(Node * v ){
  
  a[0] = v;
  downheap(0);
  return a[0];
}


void heapsort(Node * a[],int N){
  int k;
  construct(a,0);
  for( k= 1;k<=N;k++) insert( a[k] );
  for(k=1; k<=N; k++) a[k] = Remove();
}

int heapIndex( void * v){
  
  int i;
  for( i = 1; i <= N; i ++ ) if( a[i]->p == v ) return i;
  return 0;

}

Node * heapNode( void * v ){

  int i;
  for( i = 1; i <= N; i ++ ) if( a[i]->p == v ) return a[i];
  return NIL;

}

void  heapUpheap( void * v ){

  int i = heapIndex( v );
  if( !i ){
    fprintf(stderr, "There is no such element in heap \n");
    return;
  }
  upheap(i);

}

void  heapDownheap( void * v){

  int i = heapIndex( v );
  if( !i ){
    fprintf(stderr, "There is no such element in heap \n");
    return;
  }
  downheap(i);



}


void heapPrint(){

   int i;
   for( i = 1; i <=N; i++ )

     printf("root %f left-child %f right-child %f\n", a[i]->v,
	    (i*2>N)?-1:a[i*2]->v, (i*2+1>N)?-1:a[i*2+1]->v);

   
}

void heapConstruct( Node * a[],int N){
int k;
  construct(a,0);
  for( k= 1;k<=N;k++) insert( a[k] );
}

/* if new value > old value  downheap , if new value < old value, up heap */




Node * heapSelectMin(){


  return Remove();

}



void heapCheck(){
  int k,flag;

  flag = 1;
  for( k= 1;k<=N;k++){
    if( 2*k   <=N) flag &= (a[k]->v <= a[2*k  ]->v);
    if( 2*k+1 <=N) flag &= (a[k]->v <= a[2*k+1]->v);
    if( !flag ){
      printf("Error in heapCheck");
      return;
    } 
  }
  printf("good in heapCheck" );
}


int heapEmpty(){

  return (N<1) ;

}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"

Loop *LoopNew( ){
  Loop * l;

  NEW(l,Loop);
  l->ledges = NIL;
  l->lface   = NIL;
  l->alivel  = TRUE;

  return l;
}


void LoopDelete( Loop * * loop ){


   FREE(*loop);


}

/*
 *  here vertex a, b, c should be ccw
 *
 *
 */
void LoopConstruct( Face ** face, Vertex * a, Vertex * b,Vertex * c ){
 
     Loop * l = LoopNew( );
     assert(l);

     HalfEdgeConstruct(&l, a);
     HalfEdgeConstruct(&l, b);
     HalfEdgeConstruct(&l, c);

     l->lface = *face;
     (*face)->floop = l;     
}

void LoopDestruct( Loop * * loop ){
 

  
	  HalfEdgeDestruct(&((*loop)->ledges));
	  HalfEdgeDestruct(&((*loop)->ledges));
          HalfEdgeDestruct(&((*loop)->ledges));
          (*loop)->lface = NIL;
          
     LoopDelete( loop );

}












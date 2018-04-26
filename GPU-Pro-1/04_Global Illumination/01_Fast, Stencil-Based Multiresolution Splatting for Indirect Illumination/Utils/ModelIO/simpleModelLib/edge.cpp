#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"

Edge *EdgeNew( Edge * *edges ){
  Edge* e;

  NEW(e,Edge);
  ADD( (*edges), e );
  e->he1 = NIL;
  e->he2 = NIL;
  e->esolid = NIL;
  e->alive = TRUE;
  e->edgeno = 0;

  return e;
}

Edge *EdgeAddExisting( Edge *newEdge, Edge * *edges ){
  Edge* e = newEdge;

  ADD( (*edges), e );
  e->he1 = NIL;
  e->he2 = NIL;
  e->esolid = NIL;
  e->alive = TRUE;

  return e;
}


void EdgeDelete( Edge * *e ){


	DELETE((*e)->esolid->sedges,(*e));


}

/*
 *  here vertex a, b, c should be ccw
 *
 *
 */
void EdgeConstruct( Solid * * solid, HalfEdge * he1, HalfEdge *he2 ){

	  Edge * e = EdgeNew( &((*solid)->sedges) );
	  e->he1 = he1;
	  e->he2 = he2;
	  if (he1) he1->hedge = e;
	  if (he2) he2->hedge = e;
	  e->esolid = *solid;
}

void EdgeDestruct( Edge **e ){

	  assert( (*e)->he1 == NIL );
	  assert( (*e)->he2 == NIL );
	  EdgeDelete( e );

}

Face * EdgeFirstFace( Edge * e ){
  return e->he1->hloop->lface;
}
Face * EdgeSecondFace( Edge *e ){
  return e->he2->hloop->lface;
}

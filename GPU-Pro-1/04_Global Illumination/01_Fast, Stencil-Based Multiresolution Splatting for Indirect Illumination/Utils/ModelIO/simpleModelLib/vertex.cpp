#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"
 
static Id ID = 0;

void  VertexIDReset(){

  ID = 0;

}

Vertex *VertexNew( Vertex ** vertexs ){

  Vertex * v;

  NEW(v,Vertex);
  assert(v);

  v->vedge   = NIL;
  v->vertexno = ID++;
  v->alivev = TRUE;

  ADD( (*vertexs), v );

  return v;
}

Vertex *VertexAddExisting( Vertex *newVert, Vertex ** vertexs ){

  Vertex * v = newVert;

  v->vedge   = NIL;
  v->alivev = TRUE;

  ADD( (*vertexs), v );

  return v;
}


void VertexDelete(  Vertex * *v ){

       Solid *solid = (*v)->vedge->hloop->lface->fsolid;

	DELETE(solid->sverts,(*v));


}

/*
 *  here vertex a, b, c should be ccw
 *
 *
 */
void VertexConstruct( Solid * * solid, double x, double y, double z ){
 
     Vertex * v = VertexNew( &((*solid)->sverts) );
	  v->vcoord[0] = x;
	  v->vcoord[1] = y;
	  v->vcoord[2] = z;
}
void VertexConstructN( Solid * * solid, double x, double y, double z,double nx, double ny, double nz ){
 
     Vertex * v = VertexNew( &((*solid)->sverts) );
	  v->vcoord[0] = x;
	  v->vcoord[1] = y;
	  v->vcoord[2] = z;
	  v->ncoord[0] = nx;
	  v->ncoord[1] = ny;
	  v->ncoord[2] = nz;
}

void VertexDestruct( Vertex **v ){
 
     
     VertexDelete( v );

}

//-new routine

HalfEdge * VertexFirstOutHalfEdge( Vertex * v ){
     assert( v->vedge );
     return v->vedge;
}

           
HalfEdge * VertexNextOutHalfEdge( HalfEdge * he ){
   HalfEdge * mate;
        
   mate = HalfEdgeMate( he->prev );
   return mate;

}

HalfEdge * VertexFirstInHalfEdge( Vertex * v ){
     HalfEdge * mate ;
     assert( v->vedge );
     mate = HalfEdgeMate( v->vedge );
     return mate;
}


HalfEdge * VertexNextInHalfEdge( HalfEdge * he ){
   HalfEdge * mate;
   mate = HalfEdgeMate( he );
   return mate->prev;
}

Face *  VertexFirstFace( Vertex * v ){
  
  return v->vedge->hloop->lface;

}

Face * VertexNextFace( Vertex * v , Face * f ){

  HalfEdge * he = f->floop->ledges;

  do{

    if( he->hvert == v ) break;
    he = he->next;

  }while( he != f->floop->ledges );
  assert( he->hvert == v );

  he = VertexNextOutHalfEdge( he );
  return he->hloop->lface;



}
void VertexCheckConsistency( Vertex *v ){

  HalfEdge *head, *  he;
  

  he = VertexFirstOutHalfEdge( v );
  head = he;
  do{

    assert(he->aliveh);
    he = VertexNextOutHalfEdge( he );
  }while( he != head );



  he = VertexFirstInHalfEdge( v );
  head = he;
  do{

    assert(he->aliveh);
    he = VertexNextInHalfEdge( he );
  }while( he != head );


}



int VertexCheckNeighorConvexity( Vertex * end ){

  Vertex * v;
  Solid  * s;
  Face   * f, * hf;



  s = end->vedge->hloop->lface->fsolid;

  v = s->sverts;
  hf = VertexFirstFace( end );
  f  = hf;
  
  do{
    assert( f->alivef );
  do{

    if( v->alivev ){

      if( ! FaceOrientation(f, v->vcoord[0], 
			       v->vcoord[1], 
			       v->vcoord[2])) {
	printf("Concave\n");
	return 0;
      }
     
    }
   v = v->next;
  }while( v!= s->sverts );


  f = VertexNextFace(end,f);

  }while(f!=hf);

  return 1;

}


Vertex * VertexFirstVertex( Vertex * anchor ){
  
  return HalfEdgeEndVertex( anchor->vedge );
}
  
Vertex * VertexNextVertex( Vertex * anchor, Vertex * neighbor ){
  HalfEdge * head, * he, * nhe;
  
  head = anchor->vedge;
  he   = head;
  do{

    nhe = VertexNextOutHalfEdge( he );
    if( HalfEdgeEndVertex( he ) == neighbor )
        return HalfEdgeEndVertex( nhe );
    he = nhe;

  }while( he != head );
  return NIL;

}
  

/*------------------------------------------------------------------------

  Check how many adjacent faces of vert

  For manifold structure, each vertex at least has 3 adjacent faces

------------------------------------------------------------------------*/

int VertexFaceNumber( Vertex * vert ){

  Face   * f, * hf;
  int    count = 0;

  hf = VertexFirstFace( vert );
  f  = hf;
  
  do{
    assert( f->alivef );
    f = VertexNextFace( vert,f);
    count ++ ;
    
  }while(f!=hf);
  
  return count;

}

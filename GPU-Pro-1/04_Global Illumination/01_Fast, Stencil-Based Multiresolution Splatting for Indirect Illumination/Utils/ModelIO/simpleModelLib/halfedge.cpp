#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"

HalfEdge *HalfEdgeNew( HalfEdge * *halfedges ){
  HalfEdge * h;

  NEW(h,HalfEdge);
  ADD( (*halfedges), h );

  h->hedge = NIL;
  h->hloop = NIL;
  h->hvert = NIL;

  h->aliveh   = TRUE;

  return h;
}


void HalfEdgeDelete( HalfEdge * *he ){

  
   DELETE((*he)->hloop->ledges,(*he));


}

/*
 *  here vertex a, b, c should be ccw
 *
 *
 */
void HalfEdgeConstruct( Loop * * loop, Vertex *v ){
 
     HalfEdge * he = HalfEdgeNew( &((*loop)->ledges) );
     he->hvert = v;
     he->hloop = (*loop);
     v->vedge  = he;
}

void HalfEdgeDestruct( HalfEdge **he ){
 
     
     HalfEdgeDelete( he );

}

HalfEdge * HalfEdgeMate( HalfEdge *he ){

  Face *face, *tf;
  Vertex * end;
  HalfEdge * the; 
 
  if( he->hedge ) return (he->hedge->he1 != he )?he->hedge->he1:he->hedge->he2;


  end = he->next->hvert;

  face = he->hloop->lface->fsolid->sfaces;
  tf   = face;

  do{

   the = tf->floop->ledges;
   do{
   if( the->hvert == end && the->next->hvert == he->hvert ) return the;
   the = the->next;
   }while( the != tf->floop->ledges );

  tf = tf->next;
  }while( tf != face );

  // assert(0);  // This means we found an edge with only 1 triangle!!
  return NIL;

}
//new routines
Vertex * HalfEdgeStartVertex( HalfEdge * he ){
  return he->hvert;
}
Vertex * HalfEdgeEndVertex( HalfEdge * he ){
  HalfEdge * mate;
  mate = HalfEdgeMate( he );
  return mate->hvert;
}

/*--------------------------------------------------------------------------

  Merge a halfEdge
  if it succeeds, return 1;
  else return 0;

--------------------------------------------------------------------------*/
int HalfEdgeMerge( HalfEdge * he ){




 HalfEdge * the;

 Vertex   * end;
 Vertex   * he_vnext;
 Vertex   * mate_vnext;

 HalfEdge * he_next      = he->next;
 HalfEdge * he_prev      = he->prev; 
 HalfEdge * mate         = HalfEdgeMate( he );
 HalfEdge * mate_next    = mate->next;
 HalfEdge * mate_prev    = mate->prev;

 //judge if he is mergable

 if( !HalfEdgeMergable( he ) ){
   fprintf( stderr,"HalfEdgeMerge::Error halfedge is not mergable \n");
   return 0;
 }


 //disable vertex
 he->hvert->alivev = FALSE;

 //disable half edges
 he->aliveh      = FALSE;
 he_next->aliveh = FALSE;
 he_prev->aliveh = FALSE;
 mate->aliveh    = FALSE;
 mate_next->aliveh = FALSE;
 mate_prev->aliveh = FALSE;

 //disable edges

 he->hedge->alive        = FALSE;
 he_next->hedge->alive   = FALSE;
 mate_next->hedge->alive = FALSE;

 //disable faces

 he->hloop->lface->alivef   = FALSE;
 mate->hloop->lface->alivef = FALSE;


 //merge vertices

 end = HalfEdgeEndVertex( he );
 the = end->vedge;
 while( ! the->aliveh  ){  
   the = VertexNextOutHalfEdge( the );
   if( the == end->vedge ) break;
 }
 end->vedge = the;
 assert(end->vedge->aliveh);


 he_vnext = HalfEdgeEndVertex( he_next );
 the = he_vnext->vedge;
 while( ! the->aliveh  ){  
   the = VertexNextOutHalfEdge( the );
   if( the == he_vnext->vedge ) break;
 }
 he_vnext->vedge = the;
 assert(end->vedge->aliveh);
 

 mate_vnext = HalfEdgeEndVertex( mate_next );
 the = mate_vnext->vedge;
 while( ! the->aliveh  ){  
   the = VertexNextOutHalfEdge( the );
   if( the == mate_vnext->vedge ) break;
 }
 mate_vnext->vedge = the;
 assert(end->vedge->aliveh);
 




 

 the = he;
 do{ 
 if( the->aliveh )
 the->hvert = end;
 the = VertexNextOutHalfEdge( the );
 }while( the != he );


 //merge half edges
 
     if( he_prev->hedge->he1 == he_prev ){
       the = HalfEdgeMate( he_next );
       he_prev->hedge->he1 = the;
       the->hedge = he_prev->hedge;
     }
     else{
       the = HalfEdgeMate( he_next ); 
       he_prev->hedge->he2 = the;
       the->hedge = he_prev->hedge;
     }

 if( mate_prev->hedge->he1 == mate_prev ){
     the =  HalfEdgeMate( mate_next );
     mate_prev->hedge->he1 = the;
     the->hedge = mate_prev->hedge;
 }
 else{
     the = HalfEdgeMate( mate_next ); 
     mate_prev->hedge->he2 = the;
     the->hedge = mate_prev->hedge;
 }


 VertexCheckConsistency( end );

 //delete some edges

 return 1;
     
}


/*-------------------------------------------------------------------*/



void HalfEdgeExtend( HalfEdge * he ){

 HalfEdge * the;

 Vertex   * end;

 HalfEdge * he_next      = he->next;
 HalfEdge * he_prev      = he->prev; 
 HalfEdge * mate         = HalfEdgeMate( he );
 HalfEdge * mate_next    = mate->next;
 HalfEdge * mate_prev    = mate->prev;



 //Enable vertex
 he->hvert->alivev = TRUE;

 //Enable half edges
 he->aliveh      = TRUE;
 he_next->aliveh = TRUE;
 he_prev->aliveh = TRUE;
 mate->aliveh    = TRUE;
 mate_next->aliveh = TRUE;
 mate_prev->aliveh = TRUE;

 //Enable edges

 he->hedge->alive        = TRUE;
 he_next->hedge->alive   = TRUE;
 mate_next->hedge->alive = TRUE;

 //Enable faces

 he->hloop->lface->alivef   = TRUE;
 mate->hloop->lface->alivef = TRUE;

 

 end = mate->hvert;  // a bug is fixed here

 //merge half edges
 
     if( he_prev->hedge->he1->hvert ==  end ){
       the = he_prev->hedge->he2;
       he_prev->hedge->he2 = he_prev;
       the->hedge = he_next->hedge;
     }
     else{
       the = he_prev->hedge->he1;
       he_prev->hedge->he1 = he_prev;
       the->hedge = he_next->hedge;  
     }

     if( mate_prev->hedge->he1->hvert ==  end ){
       the = mate_prev->hedge->he2;
       mate_prev->hedge->he2 = mate_prev;
       the->hedge = mate_next->hedge;
     }
     else{
       the = mate_prev->hedge->he1;
       mate_prev->hedge->he1 = mate_prev;
       the->hedge = mate_next->hedge;  
     }


// VertexCheckConsistency( end );

 //delete some edges



 //merge vertices

 

 the = he;
 do{ 
 the->hvert = he->hvert;
 the = VertexNextOutHalfEdge( the );
 }while( the != he );



}


double  HalfEdgeCost( HalfEdge * he , double ( *FaceCost )( Face *f ) ){
  Vertex * end;
  int i = 0;
  Face   * hf, *f;
  double cost = 0;
  assert( he->aliveh );

    end = HalfEdgeEndVertex( he );

    /*
      HalfEdgeMerge( he );
      
      if( !VertexCheckNeighorConvexity( end ) ){
      HalfEdgeExtend( he );
      return 9999;
      }
      */

      hf = VertexFirstFace( end );
      f  = hf;
      do{
	assert( f->alivef );
	cost += FaceCost(f);
	i ++ ;
	f = VertexNextFace( end, f);
       
      }while( f!= hf );
      /*
	HalfEdgeExtend( he );
	*/
      return cost/(double)i;
    
}


/*--------------------------------------------------------------------------

  judge if halfedge is mergable

  condition

  1. after merging, two adjacent faces will collapse
  2. the vertices of these two faces will change adjacent faces
  3. calculate the number of adjacent faces of merged vertices
  4. if any of them less than three return false

---------------------------------------------------------------------------*/

int   HalfEdgeMergable2( HalfEdge * he){


  Node     * list = NIL;
  Node     * list_start = NIL;
  Node     * list_end = NIL;
  Node     * node;
  Node     * vnode;
  Vertex   * start; 
  Vertex   * end;
  Vertex   * he_next_vend;
  Vertex   * mate_next_vend;
  Vertex   * vhead;
  Vertex   * vert ;
  
  Face     * hf, *tf;
  int        l;

  HalfEdge * he_next      = he->next;
  HalfEdge * mate         = HalfEdgeMate( he );
  HalfEdge * mate_next    = mate->next;
  


  start = HalfEdgeStartVertex( he );
  end   = HalfEdgeEndVertex( he   );
  
  he_next_vend = HalfEdgeEndVertex( he_next );
  mate_next_vend = HalfEdgeEndVertex( mate_next );
  
  
  hf = VertexFirstFace( start );
  tf = hf;
  do{
    
    ListInsertNode( &list, (void*) tf, 0 );
    tf = VertexNextFace( start, tf );
  }while( tf != hf );

  hf = VertexFirstFace( end );
  tf = hf;
  do{
    
    ListInsertNode( &list, (void*) tf, 0 );
    tf = VertexNextFace( end, tf );
  }while( tf != hf );


  node = list;
  l = 0;
  do{
    l ++;
    node = node->next;
  }while( node != list );

  ListDestruct( &list );

 if( VertexFaceNumber( he_next_vend ) - 1 < 3 ) return 0;
 if( VertexFaceNumber( mate_next_vend ) - 1 < 3 ) return 0;
 if( l - 2 < 3 ) return 0;

 /*----------------------------------------------------------------------

   Test if start and end has more common neighbors other than
   he_next_vend , mate_next_vend

   -------------------------------------------------------------------*/

 
 vhead = VertexFirstVertex( start );
 vert  = vhead;

 do{
    ListInsertNode( &list_start, (void*) vert, 0 );
    vert = VertexNextVertex( start, vert );
 }while( vert != vhead );


 vhead = VertexFirstVertex( end );
 vert  = vhead;

 do{
    ListInsertNode( &list_end, (void*) vert, 0 );
    vert = VertexNextVertex( end, vert );
 }while( vert != vhead );

 node = list_start;
 do{
   vnode = list_end;
   do{

     if( node->p == vnode->p && 
	 node->p != (void*)he_next_vend && 
	 node->p != (void*)mate_next_vend )
       return 0;
     vnode = vnode->next;
   }while( vnode != list_end );
 node = node->next;
 }while( node != list_start );
 
  ListDestruct( &list_start );
  ListDestruct( &list_end );

 return 1;

}




/*--------------------------------------------------------------------------

  judge if halfedge is mergable

  condition

  1. after merging, two adjacent faces will collapse
  2. the vertices of these two faces will change adjacent faces
  3. calculate the number of adjacent faces of merged vertices
  4. if any of them less than three return false

---------------------------------------------------------------------------*/

int   HalfEdgeMergable( HalfEdge * halfedge){


  Node     * hlist_start = NIL;
  Node     * hlist_end   = NIL;
  Node     * list_start = NIL;
  Node     * list_end = NIL;

  Node     * node;
  Node     * vnode;
  Vertex   * start; 
  Vertex   * end;
  Vertex   * he_next_vend;
  Vertex   * mate_next_vend;
  Vertex   * vhead;
  Vertex   * vert ;
  
//  Face     * hf, *tf;
//  int        l;

  HalfEdge * heade, *he;
  Edge     * edge;


  HalfEdge * he_next      = halfedge->next;
  HalfEdge * mate         = HalfEdgeMate( halfedge );
  HalfEdge * mate_next    = mate->next;
  


  start = HalfEdgeStartVertex( halfedge );
  end   = HalfEdgeEndVertex( halfedge   );
  

  he_next_vend = HalfEdgeEndVertex( he_next );
  mate_next_vend = HalfEdgeEndVertex( mate_next );
  
  
  heade = VertexFirstOutHalfEdge( start );
  he    = heade;

  do{
    
    edge = he->next->hedge;
    ListInsertNode( &hlist_start, (void*)edge , 0 );
    he = VertexNextOutHalfEdge( he );
  }while( he != heade );


  heade = VertexFirstOutHalfEdge( end );
  he    = heade;

  do{
    
    edge = he->next->hedge;
    ListInsertNode( &hlist_end, (void*)edge , 0 );
    he = VertexNextOutHalfEdge( he );
  }while( he != heade );


 node = hlist_start;
 do{
   vnode = hlist_end;
   do{

     if( node->p == vnode->p )
       return 0;
     vnode = vnode->next;
   }while( vnode != hlist_end );
 node = node->next;
 }while( node != hlist_start );
 


  ListDestruct( &hlist_start );
  ListDestruct( &hlist_end   );


 /*----------------------------------------------------------------------

   Test if start and end has more common neighbors other than
   he_next_vend , mate_next_vend

   -------------------------------------------------------------------*/

 
 vhead = VertexFirstVertex( start );
 vert  = vhead;

 do{
    ListInsertNode( &list_start, (void*) vert, 0 );
    vert = VertexNextVertex( start, vert );
 }while( vert != vhead );


 vhead = VertexFirstVertex( end );
 vert  = vhead;

 do{
    ListInsertNode( &list_end, (void*) vert, 0 );
    vert = VertexNextVertex( end, vert );
 }while( vert != vhead );

 node = list_start;
 do{
   vnode = list_end;
   do{

     if( node->p == vnode->p && 
	 node->p != (void*)he_next_vend && 
	 node->p != (void*)mate_next_vend )
       return 0;
     vnode = vnode->next;
   }while( vnode != list_end );
 node = node->next;
 }while( node != list_start );
 
  ListDestruct( &list_start );
  ListDestruct( &list_end );

 return 1;

}






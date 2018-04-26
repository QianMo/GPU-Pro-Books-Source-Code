#ifndef __FUNCS_H
#define __FUNCS_H
/*ar rvf libmesh.a edge.o face.o edgelist.o facelist.o halfedge.o loop.o solid.o vertex.o vertexlist.o
ranlib libmesh.a
*/


Solid *LoadHalfEdgeModel( char *FileName );
Solid *ConstructHalfEdge_ModelFromOBJ( char *FileName );
Solid *ConstructHalfEdge_ModelFromOBJ_WStatus( char *FileName );
int OutputHalfEdgeModel( char *filename, Solid *s );


Vertex *VertexAddExisting( Vertex *newVert, Vertex ** vertexs );
Edge *EdgeAddExisting( Edge *newEdge, Edge * *edges );
Face *FaceAddExisting( Face *newFace, Face * *faces );

void  VertexIDReset();
void VertexConstruct( Solid ** , double , double , double  );
void VertexDestruct( Vertex ** );
HalfEdge * VertexFirstOutHalfEdge( Vertex * );
HalfEdge * VertexNextOutHalfEdge( HalfEdge * );
HalfEdge * VertexFirstInHalfEdge( Vertex * );
HalfEdge * VertexNextInHalfEdge( HalfEdge * );
HalfEdge * HalfEdgeMate( HalfEdge * );
Face *  VertexFirstFace( Vertex * v );
Face *  VertexNextFace( Vertex * v , Face * f );
int     VertexCheckNeighorConvexity( Vertex * );
Vertex * VertexFirstVertex( Vertex * );
Vertex * VertexNextVertex( Vertex * , Vertex * );
void VertexConstructN( Solid ** , double , double , double ,double , double , double );
void VertexCheckConsistency( Vertex * );

/*----------------------------------------------------------------------

  for checking manifold structure

---------------------------------------------------------------------*/
int VertexFaceNumber( Vertex * );



void HalfEdgeConstruct( Loop **, Vertex * );
void HalfEdgeDestruct( HalfEdge ** );
HalfEdge * HalfEdgeMate( HalfEdge * );
Vertex * HalfEdgeStartVertex( HalfEdge * );
Vertex * HalfEdgeEndVertex( HalfEdge * );
int      HalfEdgeMerge( HalfEdge * );
void     HalfEdgeExtend( HalfEdge * );
double   HalfEdgeCost( HalfEdge *  , double ( * )( Face * ) );
Vertex * HalfEdgeEndVertex( HalfEdge *  );
Vertex * HalfEdgeStartVertex( HalfEdge *  );
/*---------------------------------------------------------------------
  
  checking if current halfedge is mergable

----------------------------------------------------------------------*/
int   HalfEdgeMergable( HalfEdge * );


void EdgeConstruct( Solid **, HalfEdge *, HalfEdge *);
void  EdgeListConstructVerbose(Solid ** solid, int numFaces );
void EdgeDestruct( Edge **);
Face * EdgeFirstFace( Edge * );
Face * EdgeSecondFace( Edge *);

void FaceConstruct( Solid **, Vertex *, Vertex *,Vertex *);
void FaceDestruct( Face ** );
int FaceOrientation( Face * , double, double, double  );
int FaceOrientation2( Face * , double, double, double, double  );
void FacePrint( Face * );
void FaceNormal( Face * );

void LoopConstruct( Face **, Vertex *, Vertex *,Vertex *);
void LoopDestruct( Loop ** );

void    VertexListConstruct( Solid **,int, FILE *);
void    VertexListDestruct( Solid ** );
Vertex *VertexListIndex(Solid * ,int );
void  VertexListConstructNoff(Solid **,int,FILE *);

void  EdgeListConstruct(Solid ** );
void  EdgeListDestruct(Solid ** );

void  FaceListConstruct(Solid ** ,int ,FILE * );
void  FaceListDestruct( Solid ** );
void  FaceListOutput( Face * );

void SolidConstruct( Solid ** , char * );
void SolidCenter( Solid * s );
void SolidDestruct( Solid ** );
int  SolidConvexity( Solid * s );
void SolidConstructNoff( Solid **  , char *);


int  ListInsertNode( Node **, void *, int );
int  ListDeleteNode( Node **, void *, int );
void ListDestruct(Node **);

void downheap( int );
void upheap( int );
Node * Remove();
void heapsort(Node **,int);
int  heapIndex( Node * v);
void heapConstruct( Node **,int);
void heapPrint();
Node * heapSelectMin();
Node * heapNode( void * );
void  heapUpheap( void *);
void  heapDownheap(void *);
void  heapCheck();
int   heapEmpty(); 
double  Volumed( Face * f, double x, double y, double z );
#endif

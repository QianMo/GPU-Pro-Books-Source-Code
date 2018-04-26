#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"

void  FaceListConstruct(Solid ** solid ,int face_number,FILE *fp)
{
  int i;
  int face_size, a, b, c;
  Vertex * va, *vb, *vc;
  char Line[1024];

  for( i = 0; i < face_number && !feof(fp); i ++ ){

		 fgets(Line,1024,fp);
		 sscanf(Line,"%d %d %d %d", &face_size, &a,&b,&c);
		 va = VertexListIndex( *solid, a );
		 vb = VertexListIndex( *solid, b );
		 vc = VertexListIndex( *solid, c );
		 FaceConstruct( solid, va, vb, vc);

		}

}


void  FaceListDestruct(Solid ** solid )
{
	 Face * tf;


	 while( (*solid)->sfaces ){
	 tf = (*solid)->sfaces;
	 FaceDestruct(&tf);
	 }

}



void FaceListOutput(Face * head){

  Face * tf;
  HalfEdge *he;

  if(! head ) return;
  tf = head;
  do{
  printf("\nFace ");
  he =  tf->floop->ledges;
  assert(he);
  do{
  printf("%d ",he->hvert->vertexno);
  he = he->next;
  }while( he != tf->floop->ledges); 

  tf = tf->next;
  }while( tf != head );


}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"

void  VertexListConstruct(Solid ** solid ,int vertex_number,FILE *fp)
{
  int i;
  double x,y,z;
  char Line[1024];
  VertexIDReset();
  for( i = 0; i < vertex_number && !feof(fp); i ++ ){

		 fgets(Line,1024,fp);
		 sscanf(Line,"%lf %lf %lf", &x,&y,&z);
		 VertexConstruct(solid,x,y,z);

		}

}
void  VertexListConstructNoff(Solid ** solid ,int vertex_number,FILE *fp)
{
  int i;
  double x,y,z,nx,ny,nz;
  char Line[1024];
  VertexIDReset();
  for( i = 0; i < vertex_number && !feof(fp); i ++ ){

		 fgets(Line,1024,fp);
		 sscanf(Line,"%lf %lf %lf %lf %lf %lf", &x,&y,&z,&nx,&ny,&nz);
		 VertexConstructN(solid,x,y,z,nx,ny,nz);

		}

}

Vertex *VertexListIndex(Solid * solid, int no)
{  Vertex * tv;

	tv = solid->sverts;
	do{
	 if( tv->vertexno == no ) return tv;
	tv = tv->next;
	}while( tv != solid->sverts );
	return NIL;

}



void  VertexListDestruct(Solid ** solid )
{
	 Vertex * tv;


	 while( (*solid)->sverts ){
	 tv = (*solid)->sverts;
	 VertexDestruct( &tv);
	 }

}


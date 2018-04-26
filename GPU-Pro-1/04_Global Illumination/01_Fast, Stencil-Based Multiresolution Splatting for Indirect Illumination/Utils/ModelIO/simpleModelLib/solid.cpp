#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"

Solid *SolidNew(){
  Solid * s;

  NEW(s,Solid);
  assert(s);
  
  s->sfaces = NIL;
  s->sedges = NIL;
  s->sverts = NIL;

  return s;
}


void SolidDelete( Solid * solid  ){

  
   free(solid);


}

/*
 *  here vertex a, b, c should be ccw
 *
 *
 */
void SolidConstruct( Solid * * solid , char *FileName){

     int vertex_number, face_number, edge_number;
     char file_type[64];
     Solid *s ;

     FILE * fp = fopen(FileName,"r");
     char Line[128];
     assert(fp);

     fgets(Line,128,fp);
     sscanf(Line,"%s",file_type);
     
     fgets(Line,128,fp);
     sscanf(Line,"%d %d %d", &vertex_number,&face_number, &edge_number);
    

      s = SolidNew( );

     VertexListConstruct(&s,vertex_number,fp);
     FaceListConstruct(&s,face_number,fp);
     EdgeListConstruct(&s);

     *solid = s;
     
     fclose(fp);
}
/*
 *  here vertex a, b, c should be ccw
 *
 *
 */
void SolidConstructNoff( Solid * * solid , char *FileName){

     int vertex_number, face_number, edge_number;
     char file_type[64];
     Solid * s;
     FILE * fp = fopen(FileName,"r");
     char Line[128];
     assert(fp);

     fgets(Line,128,fp);
     sscanf(Line,"%s",file_type);
     
     fgets(Line,128,fp);
     sscanf(Line,"%d %d %d", &vertex_number,&face_number, &edge_number);
    

     s = SolidNew( );

     VertexListConstructNoff(&s,vertex_number,fp);
     FaceListConstruct(&s,face_number,fp);
     EdgeListConstruct(&s);

     *solid = s;
     
     fclose(fp);
}

void SolidDestruct( Solid * * solid ){

     FaceListDestruct( solid );
     VertexListDestruct( solid );
     EdgeListDestruct( solid );
  
     
     SolidDelete( *solid );

}



int SolidConvexity( Solid * s )
{
  register Face *f;
  register Vertex *v;
  double   vol;
  
  if ( s->sfaces->alivef ) f = s->sfaces;
  else{
    f = s->sfaces->next;
    while( !f->alivef && f != s->sfaces ) f = f->next;
    if( f == s->sfaces ) return 0;
  }
  
  
 
    do {
      v = s->sverts;
      do {
	if( v->alivev ){
	  vol = Volumed( f, v->vcoord[0], v->vcoord[1], v->vcoord[2] );
	  if ( vol < 0 ) {
	    FacePrint(f);
	    printf("%f %f %f vol %f\n",
		   v->vcoord[0], 
		   v->vcoord[1],
		   v->vcoord[2],
		   vol );
	    printf("Checks: NOT convex. \n");
	    return 0;
	  }
	  else{
	    /*
	    printf("%f %f %f vol %f\n",
		       v->vcoord[0], 
		       v->vcoord[1], 
		       v->vcoord[2],
		       vol );
	  FacePrint(f);
	  */
	  }
	  
	}
	v = v->next;
      } while ( v != s->sverts );
      f = f->next;
      
    } while ( f != s->sfaces );
  
  if ( f != s->sfaces ){
    printf( "Checks: NOT convex.\n");
    return 0;
  }
  
  printf( "Checks: convex.\n");
  return 1;
}



void SolidCenter( Solid * s ){

  Vertex * v;
  int    count;
  double  x,y,z;

  x = y = z = 0;
  count = 0;

  v = s->sverts;
  do{
  count ++;
  x += v->vcoord[0];  
  y += v->vcoord[1];  
  z += v->vcoord[2];  
  v = v->next;
  }while( v!=s->sverts);

  s->center[0] = x/(double)count;
  s->center[1] = y/(double)count;
  s->center[2] = z/(double)count;

}









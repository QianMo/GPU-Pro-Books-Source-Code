#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "mesh.h"
#include "funcs.h"

Face *FaceNew( Face * *faces ){
  static Id ID = 0;
  Face * f;

  NEW(f,Face);
  assert(f);
  
  f->faceno = ID++;
  f->floop   = NIL;
  f->fsolid = NIL;
 
  f->next   = NIL;
  f->prev   = NIL;
  
  f->alivef = TRUE;

  ADD( (*faces), f );

  return f;
}

Face *FaceAddExisting( Face *newFace, Face * *faces ){
  Face * f = newFace;

  f->floop   = NIL;
  f->fsolid = NIL;
 
  f->next   = NIL;
  f->prev   = NIL;
  
  f->alivef = TRUE;

  ADD( (*faces), f );

  return f;
}


void FaceDelete( Face * *face ){

  
   DELETE((*face)->fsolid->sfaces,(*face));


}

/*
 *  here vertex a, b, c should be ccw
 *
 *
 */
void FaceConstruct( Solid * *solid, Vertex * a, Vertex * b,Vertex * c ){
 
     Face * f = FaceNew( &((*solid)->sfaces) );

     LoopConstruct( &f,a,b,c );

     f->fsolid = *solid;

}







void FaceDestruct( Face * * face ){
 

     LoopDestruct( &((*face)->floop) );
     (*face)->fsolid = NIL;
     
     FaceDelete( face );

}
/* is the face toward the point or not */

int FaceOrientation( Face * face, double x , double y, double z ){
 double vol;

 vol = Volumed( face, x, y, z);
 
 if( vol >= 0 ) return 1;
 else return 0;

}
int FaceOrientation2( Face * face, double x , double y, double z, double threshold ){
 double vol;

 vol = Volumed( face, x, y, z);
 
 if( vol >= threshold ) return 1;
 else return 0;

}

void FacePrint( Face *f ){

  HalfEdge * he;

  he =  f->floop->ledges;
  assert(he);
  do{
    printf("%d ",he->hvert->vertexno);
    he = he->next;
  }while( he != f->floop->ledges); 
  
}

void FaceNormal( Face *f ){

  HalfEdge * he;
  int i,j;
  double   v[3][3];
  double   e[2][3];
  double   n[3];
  double   l;

  for( i = 0 ,he =  f->floop->ledges; i < 3; i ++, he = he->next )
    for( j = 0; j < 3; j ++ )
    v[i][j] = he->hvert->vcoord[j];
  
  for( j = 0; j < 3; j ++ ){
    e[0][j] = v[2][j] - v[1][j];
    e[1][j] = v[0][j] - v[1][j];
  }

  n[0] = e[0][1] * e[1][2] - e[1][1] * e[0][2];
  n[1] = e[1][0] * e[0][2] - e[0][0] * e[1][2] ;
  n[2] = e[0][0] * e[1][1] - e[1][0] * e[0][1];

  l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for( j = 0; j < 3; j ++ )
    f->normal[j] = n[j]/l;
  
}


double  Volumed( Face * f, double x, double y, double z )
{
        double  vol;
        double  ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz;

        ax = (double)f->floop->ledges->hvert->vcoord[0];
        ay = (double)f->floop->ledges->hvert->vcoord[1];
        az = (double)f->floop->ledges->hvert->vcoord[2];
        bx = (double)f->floop->ledges->next->hvert->vcoord[0];
        by = (double)f->floop->ledges->next->hvert->vcoord[1];
        bz = (double)f->floop->ledges->next->hvert->vcoord[2];
        cx = (double)f->floop->ledges->prev->hvert->vcoord[0];
        cy = (double)f->floop->ledges->prev->hvert->vcoord[1];
        cz = (double)f->floop->ledges->prev->hvert->vcoord[2];
       
        dx = (double)x;
        dy = (double)y;
        dz = (double)z;

        /* This is the expression used in the text.  Now replaced.
        vol =    -az * by * cx + ay * bz * cx + az * bx * cy - ax * bz * cy 
                - ay * bx * cz + ax * by * cz + az * by * dx - ay * bz * dx 
                - az * cy * dx + bz * cy * dx + ay * cz * dx - by * cz * dx 
                - az * bx * dy + ax * bz * dy + az * cx * dy - bz * cx * dy 
                - ax * cz * dy + bx * cz * dy + ay * bx * dz - ax * by * dz 
                - ay * cx * dz + by * cx * dz + ax * cy * dz - bx * cy * dz;
        */
        /* This expression is algebraically equivalent to the above, but
        uses fewer multiplications. */
        vol =    -(az-dz) * (by-dy) * (cx-dx) 
                + (ay-dy) * (bz-dz) * (cx-dx) 
                + (az-dz) * (bx-dx) * (cy-dy) 
                - (ax-dx) * (bz-dz) * (cy-dy) 
                - (ay-dy) * (bx-dx) * (cz-dz) 
                + (ax-dx) * (by-dy) * (cz-dz);

        return vol;
}


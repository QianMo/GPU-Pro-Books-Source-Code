#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "mesh.h"
#include "funcs.h"

#pragma warning( disable: 4996 )

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b)) 
#endif

struct _Model
{
	float *vertexPos;          // length = 3 * numVertices
	int *vertexNormIndex;      // length = numVertices
	int *vertexTexCoordIndex;  // length = numVertices

    int *triVertexIndex;       // length = 3 * numTriangles
	int *triEdgeIndex;         // length = 3 * numTriangles
	float *triNorm;            // length = 3 * numTriangles
	int *triTexCoordIndex;     // length = 3 * numTriangles

	float *normList;           // length = 3 * numVertexNormals
	float *texCoordList;       // length = 2 * numTexCoords;

	int numMaterials, maxMatlListSize;

	int numVertices;
	int numTriangles;
	int numEdges;
	int numVertexNormals;
	int numTexCoords;

	int hasVertNorms, hasVertTexCoords;

	char *file;
	char *name;
};

Solid *SolidNew();
float Unitize_Model( struct _Model *m );
void ReadOBJFirstPass( struct _Model *m, FILE* file );
void ReadOBJSecondPass( struct _Model *m, FILE* file );
void ComputeFacetNormals( struct _Model *m );
void ComputeVertexNormals( struct _Model *m );


Solid *ConstructHalfEdge_ModelFromOBJ( char *FileName )
{
	struct _Model *m;
	FILE *file;
	Solid *s;
	int i;

	/* open the file */
	file = fopen(FileName, "r");
	if (!file) return NULL;

	/* load the OBJ, unitize it, compute facet normals, etc */
	m = (struct _Model *) malloc( sizeof( struct _Model ) );
	memset( m, 0, sizeof( struct _Model ) );
	m->file = strdup( FileName );
	ReadOBJFirstPass( m, file );
	rewind(file);
	ReadOBJSecondPass( m, file );
	Unitize_Model( m );
	ComputeFacetNormals( m );
	ComputeVertexNormals( m );
	fclose( file );
    
    s = SolidNew( );
	for (i=0; i< m->numVertices; i++)
		VertexConstructN( &s, m->vertexPos[3*i+0], m->vertexPos[3*i+1], m->vertexPos[3*i+2],
							  m->normList[3*i+0],  m->normList[3*i+1],  m->normList[3*i+2] );

	for (i=0; i< m->numTriangles; i++)
	{
		 Vertex *va = VertexListIndex( s, m->triVertexIndex[3*i+0] );
		 Vertex *vb = VertexListIndex( s, m->triVertexIndex[3*i+1] );
		 Vertex *vc = VertexListIndex( s, m->triVertexIndex[3*i+2] );
		 FaceConstruct( &s, va, vb, vc);
	}

    EdgeListConstruct(&s);

    return s;
}


Solid *ConstructHalfEdge_ModelFromOBJ_WStatus( char *FileName )
{
	struct _Model *m;
	FILE *file;
	Solid *s;
	int i;

	/* open the file */
	file = fopen(FileName, "r");
	if (!file) return NULL;

	/* load the OBJ, unitize it, compute facet normals, etc */
	m = (struct _Model *) malloc( sizeof( struct _Model ) );
	memset( m, 0, sizeof( struct _Model ) );
	m->file = strdup( FileName );
	ReadOBJFirstPass( m, file );
	rewind(file);
	ReadOBJSecondPass( m, file );
	Unitize_Model( m );
	ComputeFacetNormals( m );
	ComputeVertexNormals( m );
	fclose( file );
    
    s = SolidNew( );
	for (i=0; i< m->numVertices; i++)
		VertexConstructN( &s, m->vertexPos[3*i+0], m->vertexPos[3*i+1], m->vertexPos[3*i+2],
							  m->normList[3*i+0],  m->normList[3*i+1],  m->normList[3*i+2] );

	printf("    (-) Constructing faces (%8.4f%% done)...", 0.0f ); fflush( stdout );
	for (i=0; i< m->numTriangles; i++)
	{
		 Vertex *va = VertexListIndex( s, m->triVertexIndex[3*i+0] );
		 Vertex *vb = VertexListIndex( s, m->triVertexIndex[3*i+1] );
		 Vertex *vc = VertexListIndex( s, m->triVertexIndex[3*i+2] );
		 FaceConstruct( &s, va, vb, vc);
		 if (i%1000 == 0)
		 {
			printf("\r    (-) Constructing faces (%8.4f%% done)...", 100.0f*((float)i/m->numTriangles) ); 
			fflush( stdout );
		 }
	}
	printf("\r    (-) Constructing faces (%8.4f%% done)...\n", 100.0f ); fflush( stdout );

    EdgeListConstructVerbose(&s, m->numTriangles);

    return s;
}



float Unitize_Model( struct _Model *m )
{
  int i;
  float maxx, minx, maxy, miny, maxz, minz;
  float cx, cy, cz, w, h, d;
  float scale;

  /* get the max/mins */
  maxx = minx = m->vertexPos[0];
  maxy = miny = m->vertexPos[1];
  maxz = minz = m->vertexPos[2];
  for (i = 1; i < m->numVertices; i++) {
    if (maxx < m->vertexPos[3 * i + 0])
      maxx = m->vertexPos[3 * i + 0];
    if (minx > m->vertexPos[3 * i + 0])
      minx = m->vertexPos[3 * i + 0];

    if (maxy < m->vertexPos[3 * i + 1])
      maxy = m->vertexPos[3 * i + 1];
    if (miny > m->vertexPos[3 * i + 1])
      miny = m->vertexPos[3 * i + 1];

    if (maxz < m->vertexPos[3 * i + 2])
      maxz = m->vertexPos[3 * i + 2];
    if (minz > m->vertexPos[3 * i + 2])
      minz = m->vertexPos[3 * i + 2];
  }

  /* calculate model width, height, and depth */
  w = (float)(fabs(maxx) + fabs(minx));
  h = (float)(fabs(maxy) + fabs(miny));
  d = (float)(fabs(maxz) + fabs(minz));

  /* calculate center of the model */
  cx = (maxx + minx) / 2.0f;
  cy = (maxy + miny) / 2.0f;
  cz = (maxz + minz) / 2.0f;

  /* calculate unitizing scale factor */
  scale = 2.0f / MAX(MAX(w, h), d);

  /* translate around center then scale */
  for (i = 0; i < m->numVertices; i++) {
    m->vertexPos[3 * i + 0] -= cx;
    m->vertexPos[3 * i + 1] -= cy;
    m->vertexPos[3 * i + 2] -= cz;
    m->vertexPos[3 * i + 0] *= scale;
    m->vertexPos[3 * i + 1] *= scale;
    m->vertexPos[3 * i + 2] *= scale;
  }

  return scale;
}


void ComputeFacetNormals( struct _Model *m )
{
  int  i;
  float u[3];
  float v[3];
  float tmp;
  
  /* clobber any old facetnormals */
  if (m->triNorm)
    free(m->triNorm);

  /* allocate memory for the new facet normals */
  m->triNorm = (float *)malloc( 3 * m->numTriangles * sizeof( float ) );

  for (i = 0; i < m->numTriangles; i++) 
  {
    u[0] = m->vertexPos[3 * m->triVertexIndex[3*i+1] + 0] -
           m->vertexPos[3 * m->triVertexIndex[3*i+0] + 0];
    u[1] = m->vertexPos[3 * m->triVertexIndex[3*i+1] + 1] -
           m->vertexPos[3 * m->triVertexIndex[3*i+0] + 1];
    u[2] = m->vertexPos[3 * m->triVertexIndex[3*i+1] + 2] -
           m->vertexPos[3 * m->triVertexIndex[3*i+0] + 2];

    v[0] = m->vertexPos[3 * m->triVertexIndex[3*i+2] + 0] -
           m->vertexPos[3 * m->triVertexIndex[3*i+0] + 0];
    v[1] = m->vertexPos[3 * m->triVertexIndex[3*i+2] + 1] -
           m->vertexPos[3 * m->triVertexIndex[3*i+0] + 1];
    v[2] = m->vertexPos[3 * m->triVertexIndex[3*i+2] + 2] -
           m->vertexPos[3 * m->triVertexIndex[3*i+0] + 2];

	m->triNorm[3*i+0] = u[1]*v[2] - u[2]*v[1];
    m->triNorm[3*i+1] = u[2]*v[0] - u[0]*v[2];
    m->triNorm[3*i+2] = u[0]*v[1] - u[1]*v[0];

	tmp = (float)sqrt(m->triNorm[3*i+0]*m->triNorm[3*i+0] + 
					  m->triNorm[3*i+1]*m->triNorm[3*i+1] + 
					  m->triNorm[3*i+2]*m->triNorm[3*i+2]);
	m->triNorm[3*i+0] /= tmp;
	m->triNorm[3*i+1] /= tmp;
	m->triNorm[3*i+2] /= tmp;
  }

}

void ComputeVertexNormals( struct _Model *m )
{
  int *count;
  float *partials, tmp;
  int i;

  /* nuke any previous normals */
  if (m->normList)
    free(m->normList);

  if (m->vertexNormIndex)
	  free(m->vertexNormIndex);

  /* we need facet normals for this, so make sure they're there! */
  if (!m->triNorm)
	  ComputeFacetNormals(m);

  m->numVertexNormals = m->numVertices;

  /* allocate space for new normals */
  m->normList = (float *)malloc( 3 * m->numVertexNormals * sizeof( float ) );
  m->vertexNormIndex = (int *)malloc( m->numVertexNormals * sizeof( int ) );

  /* allocate temporary space */
  partials = (float *)malloc( 3 * m->numVertexNormals * sizeof( float ) );
  count = (int *)malloc( m->numVertexNormals * sizeof( int ) );
  for (i = 0; i < m->numVertices; i++)
  {
	  m->vertexNormIndex[i] = i;
	  count[i] = 0;
	  partials[3*i+0] = partials[3*i+1] = partials[3*i+2] = 0;
  }
  
  /* for every triangle, add the facet normal to each of the vertices' partials */
  for (i = 0; i < m->numTriangles; i++) 
  {
	  partials[ 3 * m->triVertexIndex[ 3 * i + 0 ] + 0 ] += m->triNorm[ 3 * i + 0 ];
	  partials[ 3 * m->triVertexIndex[ 3 * i + 0 ] + 1 ] += m->triNorm[ 3 * i + 1 ];
	  partials[ 3 * m->triVertexIndex[ 3 * i + 0 ] + 2 ] += m->triNorm[ 3 * i + 2 ];
	  count[ m->triVertexIndex[ 3 * i + 0 ] ]++;

	  partials[ 3 * m->triVertexIndex[ 3 * i + 1 ] + 0 ] += m->triNorm[ 3 * i + 0 ];
	  partials[ 3 * m->triVertexIndex[ 3 * i + 1 ] + 1 ] += m->triNorm[ 3 * i + 1 ];
	  partials[ 3 * m->triVertexIndex[ 3 * i + 1 ] + 2 ] += m->triNorm[ 3 * i + 2 ];
	  count[ m->triVertexIndex[ 3 * i + 1 ] ]++;

	  partials[ 3 * m->triVertexIndex[ 3 * i + 2 ] + 0 ] += m->triNorm[ 3 * i + 0 ];
	  partials[ 3 * m->triVertexIndex[ 3 * i + 2 ] + 1 ] += m->triNorm[ 3 * i + 1 ];
	  partials[ 3 * m->triVertexIndex[ 3 * i + 2 ] + 2 ] += m->triNorm[ 3 * i + 2 ];
	  count[ m->triVertexIndex[ 3 * i + 2 ] ]++;
  }

  /* calculate the average normal for each vertex */
  for (i = 0; i < m->numVertices; i++) 
  {
	  m->normList[ 3 * i + 0 ] = partials[ 3 * i + 0 ] / count[i] ;
	  m->normList[ 3 * i + 1 ] = partials[ 3 * i + 1 ] / count[i] ;
	  m->normList[ 3 * i + 2 ] = partials[ 3 * i + 2 ] / count[i] ;

	  tmp = (float)sqrt(m->normList[3*i+0]*m->normList[3*i+0] + 
						m->normList[3*i+1]*m->normList[3*i+1] + 
						m->normList[3*i+2]*m->normList[3*i+2]);
	  m->normList[3*i+0] /= tmp;
	  m->normList[3*i+1] /= tmp;
	  m->normList[3*i+2] /= tmp;
  }

  free(partials);
  free(count);
}


void ReadOBJFirstPass( struct _Model *m, FILE* file ) 
{
  int    numvertices;		/* number of vertices in model */
  int    numnormals;		/* number of normals in model */
  int    numtexcoords;		/* number of texcoords in model */
  int    numtriangles;		/* number of triangles in model */
  unsigned  v, n, t;
  char      buf[1024], tmp[128], *ptr;

  numvertices = numnormals = numtexcoords = numtriangles = 0;

  while( fgets(buf, sizeof(buf), file) ) //fscanf(file, "%s", buf) != EOF) 
  {
	  switch(buf[0]) 
	  {
	  case '#':				/* comment */
		  break;
	  case 'v':				/* v, vn, vt */
		  switch(buf[1]) 	
		  {
		  case '\0':			/* vertex */
		  case ' ':
			  numvertices++;
			  break;
		  case 'n':				/* normal */
			  numnormals++;
			  break;
		  case 't':				/* texcoord */
			  numtexcoords++;
			  break;
		  default:
			  printf("ReadOBJFirstPass(): Unknown token \"%s\".\n", buf);
			  exit(1);
			  break;
		  }
		  break;
	  case 'm':            /* material file! */
		  //ptr = strstr( buf, " " );
		  //sscanf(ptr, "%s", tmp); 
		  //ReadOBJMaterialFile( m, tmp );
		  break;
	  case 'u':            /* usemtl */ 
		  break;
	  case 'g':				/* group */
		  /* there's a group.   we're probably ignoring this */
		  break;
	  case 'f':				/* face */
		  v = n = t = 0;
		  ptr = strstr( buf, " " );
		  sscanf(ptr, "%s", tmp); 
		  /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */     
		  if (strstr(tmp, "//")) 
		  {	
			  /* v//n */
			  sscanf(ptr, "%d//%d", &v, &n);
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d//%d", &v, &n);
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d//%d", &v, &n);
			  ptr = strstr( ptr+1, " " );
	          numtriangles++;
	          while(ptr && sscanf(ptr, "%d//%d", &v, &n) > 0) 
			  {
				  ptr = strstr( ptr+1, " " );
				  numtriangles++;
			  }
		  } 
		  else if (ptr && sscanf(ptr, "%d/%d/%d", &v, &t, &n) == 3) 
		  {
			  /* v/t/n */
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d/%d", &v, &t, &n);
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d/%d", &v, &t, &n);
			  ptr = strstr( ptr+1, " " );
	          numtriangles++;
	          while(ptr && sscanf(ptr, "%d/%d/%d", &v, &t, &n) > 0) 
			  {
				  ptr = strstr( ptr+1, " " );
				  numtriangles++;
			  }
		  } 
		  else if (ptr && sscanf(ptr, "%d/%d", &v, &t) == 2) 
		  {
			  /* v/t */
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d", &v, &t);
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d", &v, &t);
			  ptr = strstr( ptr+1, " " );
	          numtriangles++;
	          while(ptr && sscanf(ptr, "%d/%d", &v, &t) > 0) 
			  {
				  ptr = strstr( ptr+1, " " );
				  numtriangles++;
			  }
		  } 
		  else 
		  {
			  /* v */
			 ptr = strstr( ptr+1, " " );
	         sscanf(ptr, "%d", &v);
			 ptr = strstr( ptr+1, " " );
	         sscanf(ptr, "%d", &v);
			 ptr = strstr( ptr+1, " " );
	         numtriangles++;
	         while(ptr && sscanf(ptr, "%d", &v) > 0) 
			 {
				 ptr = strstr( ptr+1, " " );
				 numtriangles++;
			 }
		  }
		  break;
	  default:
          break;    
		  
      }
  }
		 
  /* set the stats in the model structure */
  m->numVertices        = numvertices;
  m->numVertexNormals   = numnormals;
  m->numTexCoords       = numtexcoords;
  m->numTriangles       = numtriangles;

  /* allocate memory for all the data we'll read in the second pass */
  m->vertexPos = (float *)malloc( 3 * numvertices * sizeof( float ) );
  if (numnormals > 0) m->vertexNormIndex = (int *)malloc( numvertices * sizeof( int ) );     
  if (numtexcoords > 0) m->vertexTexCoordIndex = (int *)malloc( numvertices * sizeof( int ) ); 
  m->triVertexIndex = (int *)malloc( 3 * numtriangles * sizeof( int ) );  
  if (m->numEdges > 0) m->triEdgeIndex = (int *)malloc( 3 * numtriangles * sizeof( int ) );    
  if (numtexcoords > 0) m->triTexCoordIndex = (int *)malloc( 3 * numtriangles * sizeof( int ) );
  if (numnormals > 0) m->triNorm = (float *)malloc( 3 * numtriangles * sizeof( float ) );      
  //if (m->numMaterials > 0) m->triMatl = (void **)malloc( numtriangles * sizeof( void * ) );        
  m->normList = (float *)malloc( 3 * numnormals * sizeof( float ) );         
  m->texCoordList = (float *)malloc( 2 * numtexcoords * sizeof( float ) );    
}

void ReadOBJSecondPass( struct _Model *m, FILE* file ) 
{
  int    vertCount=0;		/* number of vertices in model */
  int    normCount=0;		/* number of normals in model */
  int    texCount=0;		/* number of texcoords in model */
  int    triCount=0;		/* number of triangles in model */
  unsigned  v, n, t;
  char      buf[1024], tmp[128], *ptr;
  int    usingMaterials=0;
  void *currentMaterial=0;

  while( fgets(buf, sizeof(buf), file) ) 
  {
	  switch(buf[0]) 
	  {
	  case '#':				/* comment */
		  break;
	  case 'v':				/* v, vn, vt */
		  switch(buf[1]) 	
		  {
		  case '\0':			/* vertex */
		  case ' ':
			  sscanf( buf+1, "%f %f %f",
				  &m->vertexPos[3 * vertCount + 0], 
				  &m->vertexPos[3 * vertCount + 1], 
	              &m->vertexPos[3 * vertCount + 2]);
			  vertCount++;
			  break;
		  case 'n':				/* normal */
			  sscanf( buf+2, "%f %f %f",
				  &m->normList[3 * normCount + 0], 
				  &m->normList[3 * normCount + 1], 
	              &m->normList[3 * normCount + 2]);
			  normCount++;
			  break;
		  case 't':				/* texcoord */
			  sscanf( buf+2, "%f %f",
				  &m->texCoordList[2 * texCount + 0], 
				  &m->texCoordList[2 * texCount + 1] );
			  texCount++;
			  break;
		  default:
			  printf("ReadOBJSecondPass(): Unknown token \"%s\".\n", buf);
			  exit(1);
			  break;
		  }
		  break;
	  case 'm':
		  /* we already read the material file */

		  /* if it was successful (and we have some matls defined, use them) */
		  //if (m->materials) usingMaterials = 1;
		  break;
	  case 'u':             /* usemtl */
		  //if (!usingMaterials) break;
          //ptr = strstr( buf, " " );
		  //sscanf(ptr, "%s", tmp);
		  //currentMaterial = GetExistingMaterial( tmp ); //m->GetMaterialID( tmp );
		  //if (!currentMaterial) currentMaterial = GetExistingMaterial( "default" );
		  break;
	  case 'g':				/* group */
		  /* there's a group.   we're probably ignoring this */
		  break;
	  case 'f':				/* face */
		  v = n = t = 0;
		  ptr = strstr( buf, " " );
		  sscanf(ptr, "%s", tmp); 
		  /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */     
		  if (strstr(tmp, "//")) 
		  {	
			  /* v//n */
			  sscanf(ptr, "%d//%d", &v, &n);
			  m->triVertexIndex[3 * triCount + 0] = v-1;
			  m->triNorm[3 * triCount + 0] = n-1;   // Note!  This data is currently thrown away...
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d//%d", &v, &n);
			  m->triVertexIndex[3 * triCount + 1] = v-1;
			  m->triNorm[3 * triCount + 1] = n-1;   // Note!  This data is currently thrown away...
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d//%d", &v, &n);
			  m->triVertexIndex[3 * triCount + 2] = v-1;
			  m->triNorm[3 * triCount + 2] = n-1;   // Note!  This data is currently thrown away...
			  ptr = strstr( ptr+1, " " );
			  //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
	          triCount++;
	          while(ptr && sscanf(ptr, "%d//%d", &v, &n) > 0) 
			  {
				  m->triVertexIndex[3 * triCount + 0] = m->triVertexIndex[3 * (triCount-1) + 0];	
				  m->triNorm[3 * triCount + 0] = m->triNorm[3 * (triCount-1) + 0];
				  m->triVertexIndex[3 * triCount + 1] = m->triVertexIndex[3 * (triCount-1) + 2];	
				  m->triNorm[3 * triCount + 1] = m->triNorm[3 * (triCount-1) + 2];
				  m->triVertexIndex[3 * triCount + 2] = v-1;
				  m->triNorm[3 * triCount + 2] = n-1;  // Note!  This data is currently thrown away...
				  ptr = strstr( ptr+1, " " );
				  //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
				  triCount++;
			  }
		  } 
		  else if (ptr && sscanf(ptr, "%d/%d/%d", &v, &t, &n) == 3) 
		  {
	
			  /* v/t/n */
			  m->triVertexIndex[3 * triCount + 0] = v-1;
			  m->triTexCoordIndex[ 3 * triCount + 0] = t-1;
			  m->triNorm[3 * triCount + 0] = n-1;  // Note!  This data is currently thrown away...
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d/%d", &v, &t, &n);
			  m->triVertexIndex[3 * triCount + 1] = v-1;
			  m->triTexCoordIndex[ 3 * triCount + 1] = t-1;
			  m->triNorm[3 * triCount + 1] = n-1;  // Note!  This data is currently thrown away...
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d/%d", &v, &t, &n);
			  m->triVertexIndex[3 * triCount + 2] = v-1;
			  m->triTexCoordIndex[ 3 * triCount + 2] = t-1;
			  m->triNorm[3 * triCount + 2] = n-1;  // Note!  This data is currently thrown away...
			  ptr = strstr( ptr+1, " " );
			  //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
	          triCount++;
	          while(ptr && sscanf(ptr, "%d/%d/%d", &v, &t, &n) > 0) 
			  {
				  m->triVertexIndex[3 * triCount + 0] = m->triVertexIndex[3 * (triCount-1) + 0];	
				  m->triTexCoordIndex[ 3 * triCount + 0] = m->triTexCoordIndex[ 3 * (triCount-1) + 0];
				  m->triNorm[3 * triCount + 0] = m->triNorm[3 * (triCount-1) + 0];
				  m->triVertexIndex[3 * triCount + 1] = m->triVertexIndex[3 * (triCount-1) + 2];	
				  m->triTexCoordIndex[ 3 * triCount + 1] = m->triTexCoordIndex[ 3 * (triCount-1) + 2];
				  m->triNorm[3 * triCount + 1] = m->triNorm[3 * (triCount-1) + 2];
				  m->triVertexIndex[3 * triCount + 2] = v-1;
				  m->triTexCoordIndex[ 3 * triCount + 2] = t-1;
				  m->triNorm[3 * triCount + 2] = n-1;  // Note!  This data is currently thrown away...
				  ptr = strstr( ptr+1, " " );
				  //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
				  triCount++;
			  }
		  } 
		  else if (ptr && sscanf(ptr, "%d/%d", &v, &t) == 2) 
		  {
			  /* v/t */
			  m->triVertexIndex[3 * triCount + 0] = v-1;
			  m->triTexCoordIndex[ 3 * triCount + 0] = t-1;
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d", &v, &t);
			  m->triVertexIndex[3 * triCount + 1] = v-1;
			  m->triTexCoordIndex[ 3 * triCount + 1] = t-1;
			  ptr = strstr( ptr+1, " " );
	          sscanf(ptr, "%d/%d", &v, &t);
			  m->triVertexIndex[3 * triCount + 2] = v-1;
			  m->triTexCoordIndex[ 3 * triCount + 2] = t-1;
			  ptr = strstr( ptr+1, " " );
			  //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
	          triCount++;
	          while(ptr && sscanf(ptr, "%d/%d", &v, &t) > 0) 
			  {
				  m->triVertexIndex[3 * triCount + 0] = m->triVertexIndex[3 * (triCount-1) + 0];	
				  m->triTexCoordIndex[ 3 * triCount + 0] = m->triTexCoordIndex[ 3 * (triCount-1) + 0];
				  m->triVertexIndex[3 * triCount + 1] = m->triVertexIndex[3 * (triCount-1) + 2];	
				  m->triTexCoordIndex[ 3 * triCount + 1] = m->triTexCoordIndex[ 3 * (triCount-1) + 2];
				  m->triVertexIndex[3 * triCount + 2] = v-1;
				  m->triTexCoordIndex[ 3 * triCount + 2] = t-1;
				  ptr = strstr( ptr+1, " " );
				  //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
				  triCount++;
			  }
		  } 
		  else 
		  {
			  /* v */
			 sscanf(ptr, "%d", &v);
			 m->triVertexIndex[3 * triCount + 0] = v-1;
			 ptr = strstr( ptr+1, " " );
	         sscanf(ptr, "%d", &v);
			 m->triVertexIndex[3 * triCount + 1] = v-1;
			 ptr = strstr( ptr+1, " " );
	         sscanf(ptr, "%d", &v);
			 m->triVertexIndex[3 * triCount + 2] = v-1;
			 ptr = strstr( ptr+1, " " );
			 //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
	         triCount++;
	         while(ptr && sscanf(ptr, "%d", &v) > 0) 
			 {
				 m->triVertexIndex[3 * triCount + 0] = m->triVertexIndex[3 * (triCount-1) + 0];	
				 m->triVertexIndex[3 * triCount + 1] = m->triVertexIndex[3 * (triCount-1) + 2];	
				 m->triVertexIndex[3 * triCount + 2] = v-1;
				 ptr = strstr( ptr+1, " " );
				 //if (m->triMatl) m->triMatl[ triCount ] = currentMaterial;
				 triCount++;
			 }
		  }
		  break;
	  default:
          break;    
		  
      }
  } 
}



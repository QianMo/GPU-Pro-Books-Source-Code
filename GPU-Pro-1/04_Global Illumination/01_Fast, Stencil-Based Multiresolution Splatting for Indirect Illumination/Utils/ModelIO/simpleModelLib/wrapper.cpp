#include <stdio.h>
#include <stdlib.h>
#include "../SimpleModelLib.h"
#include "mesh.h"
#include "funcs.h"
#include <math.h>

#define BUFFER_OFFSET(x)   ((GLubyte*) NULL + (x))


HalfEdgeModel::HalfEdgeModel( char *filename, int fileType ) :
	solid(0), triList(0), edgeList(0), pointList(0), adjList(0), triVBO(0), edgeVBO(0)
{
	if (fileType == TYPE_HEM_FILE)
		solid = (void *)LoadHalfEdgeModel( filename );
	else if( fileType == TYPE_OBJ_FILE || fileType == TYPE_SMF_FILE )
		solid = (void *)ConstructHalfEdge_ModelFromOBJ_WStatus(filename);
	else
	{
		printf("***Error: Unhandled model type passed to HalfEdgeModel() constructor!\n");
	}
}


HalfEdgeModel::~HalfEdgeModel()
{
	// Get rid of the model!!
	if (solid) SolidDestruct( (Solid **)&solid );
}

void HalfEdgeModel::FreeNonGLMemory( void )
{
	if (solid) SolidDestruct( (Solid **)&solid );
	solid = 0;
}

// Output the model in our custom ".hem" file format
bool HalfEdgeModel::SaveAsHEM( char *outputFilename )
{
	// OutputHalfEdgeModel() returns non-zero on error.
	return (OutputHalfEdgeModel( outputFilename, (Solid *)solid ) == 0);
}

// Call the current display list stored in the object, if available.
bool HalfEdgeModel::CallList( int type )
{
	if (type==USE_TRIANGLES && triList > 0)				{ glCallList( triList ); return true; }
	if (type==USE_LINES && edgeList > 0)				{ glCallList( edgeList ); return true; }
	if (type==USE_POINTS && pointList > 0)				{ glCallList( pointList ); return true; }
	if (type==USE_TRIANGLE_ADJACENCY && adjList > 0)    { glCallList( adjList ); return true; }
	return false;
}

bool HalfEdgeModel::CallVBO( int type )
{
	if (type==USE_LINES && edgeVBO > 0)
	{
		glBindBuffer( GL_ARRAY_BUFFER, edgeVBO );
		glEnableClientState( GL_VERTEX_ARRAY );
		glVertexPointer( 3, GL_FLOAT, 3*sizeof(float)*vboEdgeComponents, BUFFER_OFFSET(0) );
		if (vboEdgeComponents >= 2)
		{
			glEnableClientState( GL_NORMAL_ARRAY );
			glNormalPointer( GL_FLOAT, 3*sizeof(float)*vboEdgeComponents, BUFFER_OFFSET(3*sizeof(float)) );
		}
		if (vboEdgeComponents >= 3)
		{
			glClientActiveTexture( GL_TEXTURE6 );
			glEnableClientState( GL_TEXTURE_COORD_ARRAY );
			glTexCoordPointer(3, GL_FLOAT, 3*sizeof(float)*vboEdgeComponents, BUFFER_OFFSET((vboEdgeComponents-2)*3*sizeof(float)) );
			glClientActiveTexture( GL_TEXTURE7 );
			glEnableClientState( GL_TEXTURE_COORD_ARRAY );
			glTexCoordPointer(3, GL_FLOAT, 3*sizeof(float)*vboEdgeComponents, BUFFER_OFFSET((vboEdgeComponents-1)*3*sizeof(float)) );
		}
		
		glDrawArrays( GL_LINES, 0, 2*vboEdgeCount );
	
		glDisableClientState( GL_VERTEX_ARRAY );
		if (vboEdgeComponents >= 1) glDisableClientState( GL_NORMAL_ARRAY );
		if (vboEdgeComponents >= 3)
		{
			glDisableClientState( GL_TEXTURE_COORD_ARRAY );
			glClientActiveTexture( GL_TEXTURE6 );
			glDisableClientState( GL_TEXTURE_COORD_ARRAY );
			glClientActiveTexture( GL_TEXTURE0 );
		}
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
		return true;
	}

	if (type==USE_TRIANGLES && triVBO > 0)
	{
		glBindBuffer( GL_ARRAY_BUFFER, triVBO );
		glEnableClientState( GL_VERTEX_ARRAY );
		glVertexPointer( 3, GL_FLOAT, 3*sizeof(float)*vboTriComponents, BUFFER_OFFSET(0) );
		if (vboTriComponents >= 2)
		{
			glEnableClientState( GL_NORMAL_ARRAY );
			glNormalPointer( GL_FLOAT, 3*sizeof(float)*vboTriComponents, BUFFER_OFFSET(3*sizeof(float)) );
		}
		glDrawArrays( GL_TRIANGLES, 0, 3*vboTriCount );
		glDisableClientState( GL_VERTEX_ARRAY );
		if (vboTriComponents >= 1) glDisableClientState( GL_NORMAL_ARRAY );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
		return true;
	}

	return false;
}


GLuint HalfEdgeModel::CreateOpenGLDisplayList( unsigned int flags, bool deleteOldList )
{
	// Check to see if the user wants a triangle adjacency, or just triangles
	if ( flags & USE_TRIANGLE_ADJACENCY )
	{
		if ( adjList > 0 && deleteOldList ) { glDeleteLists( adjList, 1 ); adjList = 0; }
		return CreateTriangleAdjacencyDisplayList( flags );
	}
	else if ( flags & USE_LINES )
	{
		if ( edgeList > 0 && deleteOldList ) { glDeleteLists( edgeList, 1 ); edgeList = 0; }
		return CreateEdgeDisplayList( flags );
	}
	else
	{
		if ( triList > 0 && deleteOldList ) { glDeleteLists( triList, 1 ); triList = 0; }
		return CreateTriangleDisplayList( flags );
	}
}

GLuint HalfEdgeModel::CreateOpenGLVBO( unsigned int flags )
{
	// Check to see if the user wants lines, triangles, or (as yet unsupported) triangle adjacency
	if ( flags & USE_LINES )
	{
		if ( edgeVBO > 0 ) { glDeleteBuffers( 1, &edgeVBO ); edgeVBO = 0; }
		return CreateEdgeVBO( flags );
	}
	else
	{
		if ( triVBO > 0 ) { glDeleteBuffers( 1, &triVBO ); triVBO = 0; }
		return CreateTriangleVBO( flags );
	}
}

GLuint HalfEdgeModel::CreateEdgeVBO( unsigned int flags )
{
	bool normals =    (flags & WITH_NORMALS) > 0;
	bool facetNorms = (flags & WITH_ADJACENT_FACE_NORMS) > 0;
	bool failure = false;
	float fnorm1[3], fnorm2[3];
	int edgeCount=0;
	Solid *sobj = (Solid *)solid;
	int numComponents = 1 + (normals?1:0) + (facetNorms?2:0);

	// Do this in two passes.  The first checks for validity and counts the number
	//    of edges.  Here's pass 1.
	for ( Edge *currEdge = sobj->sedges; currEdge; ) 
	{
		if( !currEdge->he1 && !currEdge->he2)
		{
			currEdge = currEdge->next;
			if (!currEdge || currEdge == sobj->sedges) break;
			continue;
		}
		if(!currEdge->he1 || !currEdge->he2)
		{
			HalfEdge *he = currEdge->he1 ? currEdge->he1 : currEdge->he2;
			if(!he->next) { failure=true; break; }
		}
		edgeCount++;
		currEdge = currEdge->next;
		if (!currEdge || currEdge == sobj->sedges) break;
	}
	if (failure)
	{
		printf("Unable to create edge VBO.  Corrupted half-edge loop!\n");
		return 0;
	}

	// Allocate enough memory for all the edges
	unsigned int dataSize = edgeCount*6*sizeof(float)*numComponents;
	float *floatData = (float *)malloc( dataSize );
	int currentEdge = 0;
	if (!floatData)
	{
		printf("Unable to allocate temporary memory for edge VBO!\n");
		return 0;
	}

	// Copy data into allocated memory
	for (Edge *currEdge = sobj->sedges; currEdge; ) 
	{
		int vertOff0 = 6*numComponents*currentEdge;
		int vertOff1 = vertOff0+3*numComponents;
		int normOff0 = vertOff0+3;
		int normOff1 = normOff0+3*numComponents;
		int fnormOff0_1 = vertOff0+3+(normals?3:0);
		int fnormOff1_1 = fnormOff0_1+3*numComponents;
		int fnormOff0_2 = fnormOff0_1+3;
		int fnormOff1_2 = fnormOff1_1+3;

		if( !currEdge->he1 && !currEdge->he2)
		{
			currEdge = currEdge->next;
			if (!currEdge || currEdge == sobj->sedges) break;
			continue;
		}
		
		if (facetNorms) 
		{
			if(!currEdge->he1)
				{ fnorm1[0] = 0; fnorm1[1] = 0; fnorm1[2] = 0;}
			else
				ComputeFaceNormal( fnorm1, (void *)(currEdge->he1->hloop->lface) );
			if(!currEdge->he2)
				{ fnorm2[0] = 0; fnorm2[1] = 0; fnorm2[2] = 0; }
			else
				ComputeFaceNormal( fnorm2, (void *)(currEdge->he2->hloop->lface) );
			floatData[fnormOff0_1+0] = fnorm1[0]; floatData[fnormOff0_1+1] = fnorm1[1]; floatData[fnormOff0_1+2] = fnorm1[2]; 
			floatData[fnormOff1_1+0] = fnorm1[0]; floatData[fnormOff1_1+1] = fnorm1[1]; floatData[fnormOff1_1+2] = fnorm1[2]; 
			floatData[fnormOff0_2+0] = fnorm2[0]; floatData[fnormOff0_2+1] = fnorm2[1]; floatData[fnormOff0_2+2] = fnorm2[2]; 
			floatData[fnormOff1_2+0] = fnorm2[0]; floatData[fnormOff1_2+1] = fnorm2[1]; floatData[fnormOff1_2+2] = fnorm2[2];
		}
		if(!currEdge->he1 || !currEdge->he2)
		{
			HalfEdge *he = currEdge->he1 ? currEdge->he1 : currEdge->he2;
			//only one halfedge was valid.
			if (normals) 
				{ floatData[normOff0+0] = he->hvert->ncoord[0]; floatData[normOff0+1] = he->hvert->ncoord[1]; floatData[normOff0+2] = he->hvert->ncoord[2]; }
			floatData[vertOff0+0] = he->hvert->vcoord[0]; floatData[vertOff0+1] = he->hvert->vcoord[1]; floatData[vertOff0+2] = he->hvert->vcoord[2]; 

			// the other halfedge is next in the loop
			he = he->next;
			if (normals) 
				{ floatData[normOff1+0] = he->hvert->ncoord[0]; floatData[normOff1+1] = he->hvert->ncoord[1]; floatData[normOff1+2] = he->hvert->ncoord[2]; }
			floatData[vertOff1+0] = he->hvert->vcoord[0]; floatData[vertOff1+1] = he->hvert->vcoord[1]; floatData[vertOff1+2] = he->hvert->vcoord[2]; 
		}
		else
		{
			HalfEdge *he = currEdge->he1;
			if (normals) 
				{ floatData[normOff0+0] = he->hvert->ncoord[0]; floatData[normOff0+1] = he->hvert->ncoord[1]; floatData[normOff0+2] = he->hvert->ncoord[2]; }
			floatData[vertOff0+0] = he->hvert->vcoord[0]; floatData[vertOff0+1] = he->hvert->vcoord[1]; floatData[vertOff0+2] = he->hvert->vcoord[2]; 

			he = currEdge->he2;
			if (normals) 
				{ floatData[normOff1+0] = he->hvert->ncoord[0]; floatData[normOff1+1] = he->hvert->ncoord[1]; floatData[normOff1+2] = he->hvert->ncoord[2]; }
			floatData[vertOff1+0] = he->hvert->vcoord[0]; floatData[vertOff1+1] = he->hvert->vcoord[1]; floatData[vertOff1+2] = he->hvert->vcoord[2]; 
		}
		
		currentEdge++;
		currEdge = currEdge->next;
		if (!currEdge || currEdge == sobj->sedges) break;
	}

	vboEdgeCount = currentEdge;
	vboEdgeComponents = numComponents;

	if (edgeVBO > 0) glDeleteBuffers( 1, &edgeVBO );
	glGenBuffers( 1, &edgeVBO );
	glBindBuffer( GL_ARRAY_BUFFER, edgeVBO );
	glBufferData( GL_ARRAY_BUFFER, dataSize, floatData, GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	free( floatData );

	return edgeVBO;
}


GLuint HalfEdgeModel::CreateTriangleVBO( unsigned int flags )
{
	bool normals =    (flags & WITH_NORMALS) > 0;
	bool failure = false;
	GLuint triCount = 0;
	Solid *sobj = (Solid *)solid;
	int numComponents = 1 + (normals?1:0);
	vboTriComponents = numComponents;

	for (Face *currFace = sobj->sfaces; currFace; ) 
	{
		Loop *loop   = currFace->floop;
		if (!loop) { failure=true; break; }
		HalfEdge *he1 = loop->ledges;
		if (!he1) { failure=true; break; }
		HalfEdge *he2 = he1->next;
		if (!he2) { failure=true; break; }
		HalfEdge *he3 = he2->next;
		if (!he3) { failure=true; break; }
		triCount++;
		currFace = currFace->next;
		if (!currFace || currFace == sobj->sfaces) break;
	}
	if (failure)
	{
		printf("Unable to create triangle VBO.  Corrupted half-edge structure!\n");
		return 0;
	}

	// Allocate enough memory for all the edges
	unsigned int dataSize = triCount*9*sizeof(float)*numComponents;
	vboTriCount = triCount;
	float *floatData = (float *)malloc( dataSize );
	int currentTri = 0;
	if (!floatData)
	{
		printf("Unable to allocate temporary memory for triangle VBO!\n");
		return 0;
	}

	// Copy data into the temporary vertex array memory
	for (Face *currFace = sobj->sfaces; currFace; ) 
	{
		int vertOff0 = 9*numComponents*currentTri;
		int vertOff1 = vertOff0+3*numComponents;
		int vertOff2 = vertOff0+6*numComponents;
		int normOff0 = 9*numComponents*currentTri+3;
		int normOff1 = normOff0+3*numComponents;
		int normOff2 = normOff0+6*numComponents;

		Loop *loop   = currFace->floop;
		HalfEdge *he1 = loop->ledges;
		HalfEdge *he2 = he1->next;
		HalfEdge *he3 = he2->next;

		if (numComponents>1) 
			{ floatData[normOff0+0] = he1->hvert->ncoord[0]; floatData[normOff0+1] = he1->hvert->ncoord[1]; floatData[normOff0+2] = he1->hvert->ncoord[2]; }
		floatData[vertOff0+0] = he1->hvert->vcoord[0]; floatData[vertOff0+1] = he1->hvert->vcoord[1]; floatData[vertOff0+2] = he1->hvert->vcoord[2];
		if (numComponents>1) 
			{ floatData[normOff1+0] = he2->hvert->ncoord[0]; floatData[normOff1+1] = he2->hvert->ncoord[1]; floatData[normOff1+2] = he2->hvert->ncoord[2]; }
		floatData[vertOff1+0] = he2->hvert->vcoord[0]; floatData[vertOff1+1] = he2->hvert->vcoord[1]; floatData[vertOff1+2] = he2->hvert->vcoord[2];
		if (numComponents>1) 			
			{ floatData[normOff2+0] = he3->hvert->ncoord[0]; floatData[normOff2+1] = he3->hvert->ncoord[1]; floatData[normOff2+2] = he3->hvert->ncoord[2]; }
		floatData[vertOff2+0] = he3->hvert->vcoord[0]; floatData[vertOff2+1] = he3->hvert->vcoord[1]; floatData[vertOff2+2] = he3->hvert->vcoord[2];

		currentTri++;
		currFace = currFace->next;
		if (!currFace || currFace == sobj->sfaces) break;
	}

	glGenBuffers( 1, &triVBO );
	glBindBuffer( GL_ARRAY_BUFFER, triVBO );
	glBufferData( GL_ARRAY_BUFFER, dataSize, floatData, GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	free( floatData );

	return triVBO;
}



bool HalfEdgeModel::ComputeFaceNormal( float *resultNorm, void *face )
{
	Face *f = (Face *)face;
	HalfEdge *h1, *h2, *h3;
	if (!f || !f->floop || !f->floop->ledges) return false;

	h2 = f->floop->ledges;
	if (!h2) return false;
	h1 = h2->prev;
	h3 = h2->next;
	if (!h1 || !h3) return false;

	// We now have 3 half edges on the given face.  We will assume
	//    the halfedge loop goes in a counter-clockwise fashion to
	//    compute a normal.
	double vec1[3], vec2[3];
	vec1[0] = h3->hvert->vcoord[0] - h2->hvert->vcoord[0];
	vec1[1] = h3->hvert->vcoord[1] - h2->hvert->vcoord[1];
	vec1[2] = h3->hvert->vcoord[2] - h2->hvert->vcoord[2];
	vec2[0] = h1->hvert->vcoord[0] - h2->hvert->vcoord[0];
	vec2[1] = h1->hvert->vcoord[1] - h2->hvert->vcoord[1];
	vec2[2] = h1->hvert->vcoord[2] - h2->hvert->vcoord[2];

	// Cross product
	resultNorm[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
	resultNorm[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
	resultNorm[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];

	// Normalize
	float lengthSqr = resultNorm[0]*resultNorm[0] + resultNorm[1]*resultNorm[1] + resultNorm[2]*resultNorm[2];
	float one_length = 1.0f / sqrtf( lengthSqr );
	resultNorm[0] *= one_length;
	resultNorm[1] *= one_length;
	resultNorm[2] *= one_length;
	return true;
}



GLuint HalfEdgeModel::CreateEdgeDisplayList( unsigned int flags )
{
	bool normals =    (flags & WITH_NORMALS) > 0;
	bool facetNorms = (flags & WITH_ADJACENT_FACE_NORMS) > 0;
	bool failure = false;
	float fnorm1[3], fnorm2[3];
	Solid *sobj = (Solid *)solid;
	GLuint list = glGenLists( 1 );
	glNewList( list, GL_COMPILE );


	glBegin( GL_LINES );
	Edge *currEdge = sobj->sedges;
	for ( ; currEdge; ) 
	{
		
		if( !currEdge->he1 && !currEdge->he2)
		{
			currEdge = currEdge->next;
			if (!currEdge || currEdge == sobj->sedges) break;
			continue;
		}
		
		if (facetNorms) 
		{
			if(!currEdge->he1)
			{ fnorm1[0] = 0; fnorm1[1] = 0; fnorm1[2] = 0;}
			else
				ComputeFaceNormal( fnorm1, (void *)(currEdge->he1->hloop->lface) );
			if(!currEdge->he2)
			{ fnorm2[0] = 0; fnorm2[1] = 0; fnorm2[2] = 0; }
			else
				ComputeFaceNormal( fnorm2, (void *)(currEdge->he2->hloop->lface) );
			glMultiTexCoord3fv( GL_TEXTURE6, fnorm1 );
			glMultiTexCoord3fv( GL_TEXTURE7, fnorm2 );
		}
		if(!currEdge->he1 || !currEdge->he2)
		{
			HalfEdge *he = currEdge->he1 ? currEdge->he1 : currEdge->he2;
			//only one halfedge was valid.
			if (normals) 
				glNormal3dv( he->hvert->ncoord );
			glVertex3dv( he->hvert->vcoord );
			// the other halfedge is next in the loop
			he = he->next;
			if(!he) 
			{
				printf("Error: Corrupted half-edge loop!\n");
				failure=true;
				break;
			}
			else
			{
				if(normals) glNormal3dv(he->hvert->ncoord);
				glVertex3dv(he->hvert->vcoord);
			}
		}
		else
		{
			if (normals) glNormal3dv( currEdge->he1->hvert->ncoord );
			glVertex3dv( currEdge->he1->hvert->vcoord );
			if (normals) glNormal3dv( currEdge->he2->hvert->ncoord );
			glVertex3dv( currEdge->he2->hvert->vcoord );
		}
		
		currEdge = currEdge->next;
		if (!currEdge || currEdge == sobj->sedges) break;
	}
	glEnd();
	glEndList();

	if (failure)
	{
		glDeleteLists( list, 1 );
		return 0;
	}

	return (edgeList = list);
}


GLuint HalfEdgeModel::CreateTriangleDisplayList( unsigned int flags )
{
	bool normals =    (flags & WITH_NORMALS) > 0;
	bool facetNorms = (flags & WITH_FACET_NORMALS) > 0;
	bool failure = false;
	float facetNorm[3];
	Solid *sobj = (Solid *)solid;
	GLuint list = glGenLists( 1 );
	glNewList( list, GL_COMPILE );

	glBegin( GL_TRIANGLES );
	Face *currFace = sobj->sfaces;
	for ( ; currFace; ) 
	{
		Loop *loop   = currFace->floop;
		if (!loop) { failure=true; break; }
		HalfEdge *he1 = loop->ledges;
		if (!he1) { failure=true; break; }
		HalfEdge *he2 = he1->next;
		if (!he2) { failure=true; break; }
		HalfEdge *he3 = he2->next;
		if (!he3) { failure=true; break; }
		if (facetNorms)
		{
			ComputeFaceNormal( facetNorm, (void *)(loop->lface) );	
			glNormal3fv( facetNorm );
		}
		if (normals) glNormal3dv( he1->hvert->ncoord );
		glVertex3dv( he1->hvert->vcoord );
		if (normals) glNormal3dv( he2->hvert->ncoord );
		glVertex3dv( he2->hvert->vcoord );
		if (normals) glNormal3dv( he3->hvert->ncoord );
		glVertex3dv( he3->hvert->vcoord );
		currFace = currFace->next;
		if (!currFace || currFace == sobj->sfaces) break;
	}
	glEnd();
	glEndList();

	if (failure)
	{
		glDeleteLists( list, 1 );
		return 0;
	}

	return (triList = list);
}



GLuint HalfEdgeModel::CreateTriangleAdjacencyDisplayList( unsigned int flags )
{
	bool normals =    (flags & WITH_NORMALS) > 0;
	bool facetNorms = (flags & WITH_FACET_NORMALS) > 0;
	bool failure = false;
	float centralFacetNorm[3], extFacetNorm[3];
	Solid *sobj = (Solid *)solid;
	GLuint list = glGenLists( 1 );
	glNewList( list, GL_COMPILE );

	glBegin( GL_TRIANGLES_ADJACENCY_EXT );
	Face *currFace = sobj->sfaces;
	for ( ; currFace; ) 
	{
		Loop *loop   = currFace->floop;
		if (!loop) { failure=true; break; }
		HalfEdge *he1 = loop->ledges;
		if (!he1) { failure=true; break; }
		HalfEdge *he2 = he1->next;
		if (!he2) { failure=true; break; }
		HalfEdge *he3 = he2->next;
		if (!he3) { failure=true; break; }
		/* vertex 1 */
		if (facetNorms)
		{
			ComputeFaceNormal( centralFacetNorm, (void *)(loop->lface) );	
			glNormal3fv( centralFacetNorm );
		}
		if (normals) glNormal3dv( he1->hvert->ncoord );
		glVertex3dv( he1->hvert->vcoord );

		/* vertex 2 */
		HalfEdge *ext = (he1->hedge->he1 == he1) ? he1->hedge->he2 : he1->hedge->he1;
		if (ext)
		{
			ext = ext->prev;
			if (facetNorms)
			{
				ComputeFaceNormal( extFacetNorm, (void *)(ext->hloop->lface) );	
				glNormal3fv( extFacetNorm );
			}
			if (normals) glNormal3dv( ext->hvert->ncoord );
			glVertex3dv( ext->hvert->vcoord );
		}
		else
		{ 
			if (normals || facetNorms) glNormal3d( 0, 0, 0 ); 
			glVertex3d( 0, 0, 0 ); 
		}

		/* vertex 3 */
		if (facetNorms) glNormal3fv( centralFacetNorm );
		if (normals) glNormal3dv( he2->hvert->ncoord );
		glVertex3dv( he2->hvert->vcoord );

		/* vertex 4 */
		ext = (he3->hedge->he1 == he3) ? he3->hedge->he2 : he3->hedge->he1;
		if (ext)
		{
			ext = ext->prev;
			if (facetNorms)
			{
				ComputeFaceNormal( extFacetNorm, (void *)(ext->hloop->lface) );	
				glNormal3fv( extFacetNorm );
			}
			if (normals) glNormal3dv( ext->hvert->ncoord );
			glVertex3dv( ext->hvert->vcoord );
		}
		else
		{ 
			if (normals || facetNorms) glNormal3d( 0, 0, 0 ); 
			glVertex3d( 0, 0, 0 ); 
		}

		/* vertex 5 */
		if (facetNorms) glNormal3fv( centralFacetNorm );
		if (normals) glNormal3dv( he3->hvert->ncoord );
		glVertex3dv( he3->hvert->vcoord );

		/* vertex 6 */
		ext = (he2->hedge->he1 == he2) ? he2->hedge->he2 : he2->hedge->he1;
		if (ext)
		{
			ext = ext->prev;
			if (facetNorms)
			{
				ComputeFaceNormal( extFacetNorm, (void *)(ext->hloop->lface) );	
				glNormal3fv( extFacetNorm );
			}
			if (normals) glNormal3dv( ext->hvert->ncoord );
			glVertex3dv( ext->hvert->vcoord );
		}
		else
		{ 
			if (normals || facetNorms) glNormal3d( 0, 0, 0 ); 
			glVertex3d( 0, 0, 0 ); 
		}

		currFace = currFace->next;
		if (!currFace || currFace == sobj->sfaces) break;
	}
	glEnd();

	glEndList();

	if (failure)
	{
		glDeleteLists( list, 1 );
		return 0;
	}

	return (adjList = list);
}


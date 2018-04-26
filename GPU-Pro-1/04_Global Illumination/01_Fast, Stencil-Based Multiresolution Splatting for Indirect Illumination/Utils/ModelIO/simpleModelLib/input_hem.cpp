#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "mesh.h"
#include "funcs.h"

Solid *SolidNew();

Solid *LoadHalfEdgeModel( char *FileName )
{
	FILE *file;
	Solid *s;
	char buf[512];
	unsigned int vertCount=0, triCount=0, edgeCount=0;
	unsigned int verts=0, tris=0, edges=0;
	bool invertNormals=false, invertFaces=false;
	Vertex *vertMem=0;
	Edge *edgeMem=0;
	Face *faceMem=0;

	/* open the file */
	file = fopen(FileName, "r");
	if (!file) return NULL;

	// Get our edge, triangle, and vertex counts!
	while ( !vertCount || !triCount || !edgeCount )
	{
		if ( fgets(buf, 512, file) == NULL ) break;
		if (buf[0] == '#' || buf[0] == ' ') continue;
		if (buf[0] == 'i')
		{
			char *ptr = strtok( buf, " \r\n" );
			if ( !strcmp(ptr, "invert") )
			{
				ptr = strtok( NULL, " \r\n" );
				if (!strcmp(ptr, "normals" ))
					invertNormals=true;
				else if (!strcmp(ptr, "faces" ))
					invertFaces=true;
			}
		}
		if (buf[0] == 'v')
			sscanf( buf, "verts %d", &vertCount );
		if (buf[0] == 't')
			sscanf( buf, "tris  %d", &triCount );
		if (buf[0] == 'e')
			sscanf( buf, "edges %d", &edgeCount );
	}

	//printf("Count: %d %d %d, Inverted: %d %d\n", vertCount, triCount, edgeCount, invertNormals?1:0, invertFaces?1:0 );
	
	// If we got to the end of the file, print an error, return NULL
	if (feof(file))
	{
		fclose( file );
		printf("**** Error: Reached EOF before encountering vert/tri/edge count!\n");
		return 0;
	}

	vertMem = (Vertex *) malloc( sizeof( Vertex ) * vertCount );
	faceMem = (Face *) malloc( sizeof( Face ) * triCount );
	edgeMem = (Edge *) malloc( sizeof( Edge ) * edgeCount );

	s = SolidNew( );
	s->sedges = edgeMem;
	s->sfaces = faceMem;
	s->sverts = vertMem;

//#define DEBUG
#ifdef DEBUG
	printf("# edge faces = %d\n",edgeCount);
#endif

	// Read in the geometry
	while ( !feof( file ) )
	{
		if ( fgets(buf, 512, file) == NULL ) break;
		if (buf[0] == '#' || buf[0] == ' ') continue;

		if (buf[0] == 'v')
		{
			unsigned int ID;
			float pos[3], norm[3];
			sscanf( buf, "v %d %f %f %f %f %f %f", &ID,
					&pos[0], &pos[1], &pos[2], &norm[0], &norm[1], &norm[2] );
			VertexAddExisting( &vertMem[ID], &(s->sverts) );
			vertMem[ID].vertexno = ID;
			vertMem[ID].vcoord[0] = pos[0];
			vertMem[ID].vcoord[1] = pos[1];
			vertMem[ID].vcoord[2] = pos[2];
			vertMem[ID].ncoord[0] = (invertNormals?-norm[0]:norm[0]);
			vertMem[ID].ncoord[1] = (invertNormals?-norm[1]:norm[1]);
			vertMem[ID].ncoord[2] = (invertNormals?-norm[2]:norm[2]);
		}
		else if (buf[0] == 't')
		{
			HalfEdge *he;
			Face *ptr;
			unsigned int ID;
			unsigned int vertID[3], edgeID[3];
			sscanf( buf, "t %d %d %d %d %d %d %d", &ID,
					&vertID[0], &vertID[1], &vertID[2], &edgeID[0], &edgeID[1], &edgeID[2] );
			FaceAddExisting( &faceMem[ID], &(s->sfaces) );
			faceMem[ID].fsolid = s;
			faceMem[ID].faceno = ID;
			ptr = &faceMem[ID];
			if (!invertFaces)
				LoopConstruct( &ptr, &vertMem[vertID[0]], &vertMem[vertID[1]], &vertMem[vertID[2]] );
			else
				LoopConstruct( &ptr, &vertMem[vertID[0]], &vertMem[vertID[2]], &vertMem[vertID[1]] );
			he = faceMem[ID].floop->ledges;
			int curEdgeID = (invertFaces ? edgeID[2] : edgeID[0] );
			he->hedge = &edgeMem[curEdgeID];
			if (edgeMem[curEdgeID].he1)
				edgeMem[curEdgeID].he2 = he;
			else
				edgeMem[curEdgeID].he1 = he;

			he = he->next;
			curEdgeID = edgeID[1]; // (invertFaces ? edgeID[1] : edgeID[1] );
			he->hedge = &edgeMem[curEdgeID];
			if (edgeMem[curEdgeID].he1)
				edgeMem[curEdgeID].he2 = he;
			else
				edgeMem[curEdgeID].he1 = he;

			he = he->next;
			curEdgeID = (invertFaces ? edgeID[0] : edgeID[2] );
			he->hedge = &edgeMem[curEdgeID];
			if (edgeMem[curEdgeID].he1)
				edgeMem[curEdgeID].he2 = he;
			else
				edgeMem[curEdgeID].he1 = he;
		}
		else if (buf[0] == 'e')
		{
			unsigned int ID;
			unsigned int vertID1, vertID2;
			sscanf( buf, "e %d %d %d", &ID, &vertID1, &vertID2 );
		//	printf(" Read e %d %d %d\n",ID, vertID1, vertID2 ); 
			EdgeAddExisting( &edgeMem[ID], &(s->sedges) );
			edgeMem[ID].edgeno = ID;
			edgeMem[ID].esolid = s;
			edgeMem[ID].he1 = 0;
			edgeMem[ID].he2 = 0;
		}
	}

	fclose( file );
    
	// Now we have to update the edge list to make sure we point to the half edges.
    //EdgeListConstruct(&s);

    return s;
}
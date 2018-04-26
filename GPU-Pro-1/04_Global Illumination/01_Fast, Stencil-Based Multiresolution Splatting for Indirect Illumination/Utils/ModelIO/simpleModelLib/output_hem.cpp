#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#include <string.h>
#include "mesh.h"
#include "funcs.h"



int OutputHalfEdgeModel( char *filename, Solid *s )
{
	FILE *f;
	unsigned int numVerts = 0;
	unsigned int numEdges = 0;
	unsigned int numTris  = 0;
	unsigned int i;
	Face *currFace;
	Vertex *currVer;
	Edge *currEdge;

	// Make sure we can open our file before going through the
	//    trouble of getting ready to write...
	f = fopen( filename, "w" );
	if (!f) 
	{
		printf("*** Error: Unable to write to '%s'\n", filename);
		return -1;
	}

	// Count the number of faces...
	currFace = s->sfaces;
	do 
	{
		numTris++;
		currFace = currFace->next;
		if (!currFace || currFace == s->sfaces) break;
	} while(1);

	// Count the number of verts...
	currVer = s->sverts;
	do 
	{
		numVerts++;
		currVer = currVer->next;
		if (!currVer || currVer == s->sverts) break;
	} while(1);

	// Count the number of verts...
	currEdge = s->sedges;
	do 
	{
		numEdges++;
		currEdge = currEdge->next;
		if (!currEdge || currEdge == s->sedges) break;
	} while(1);

	fprintf( f, "# Half-edge model output by Chris' half-edge model library.\n");
	fprintf( f, "#         (Built on Xianfeng Gu's half-edge code)\n");
	fprintf( f, "# Format:\n");
	fprintf( f, "#    Triangle, edge, and vertex counts (3 lines)\n");
	fprintf( f, "#    Vertices  (\"v <id> <position x,y,z> <normal x,y,z>\")\n");
	fprintf( f, "#    Edges     (\"e <id> <vertID #1> <vertID #2>\")\n");
	fprintf( f, "#    Triangles (\"t <id> <vertID #1> <vertID #2> <vertID #3> <edgeID #1> <edgeID #2> <edgeID #3>\")\n");
	fprintf( f, "# For edges, vertID1 < vertID2.  For triangles, vertID #1, #2, and #3 are specified in order, with\n");
	fprintf( f, "#    edgeID #1 corresponding to the edge between vertID #1 and #2.\n");
	fprintf( f, "\ntris  %d\n", numTris );
	fprintf( f, "edges %d\n", numEdges );
	fprintf( f, "verts %d\n\n", numVerts );

	// Output vertices
	i=0;
	currVer = s->sverts;
	do 
	{
		fprintf(f, "v %d %f %f %f %f %f %f\n", 
				currVer->vertexno,
				currVer->vcoord[0], currVer->vcoord[1], currVer->vcoord[2], 
				currVer->ncoord[0], currVer->ncoord[1], currVer->ncoord[2] );
		i++;
		currVer = currVer->next;
		if (!currVer || currVer == s->sverts) break;
		if (i >= numVerts) break;
	} while(1);
	fprintf( f, "\n");

	// Output edges
	i=0;
	currEdge = s->sedges;
	do 
	{
		HalfEdge *h1 = currEdge->he1, *h2 = currEdge->he2;
		int vertID1 = -1, vertID2 = -1;
		if (h1 && h2)
		{
			vertID1 = h1->hvert->vertexno;
			vertID2 = h2->hvert->vertexno;
		}
		else if (h1)
		{
			vertID1 = h1->hvert->vertexno;
			vertID2 = h1->next->hvert->vertexno;
		}
		else if (h2)
		{
			vertID1 = h2->hvert->vertexno;
			vertID2 = h2->next->hvert->vertexno;
		}
		
		currEdge->edgeno = i;

		// Make sure the low #'d vertex is in ID1.
		if (vertID2 < vertID1)
		{
			int tmp = vertID1;
			vertID1 = vertID2;
			vertID2 = tmp;
		}

		fprintf(f, "e %d %d %d\n", 
				currEdge->edgeno, vertID1, vertID2 );
		i++;
		currEdge = currEdge->next;
		if (!currEdge || currEdge == s->sedges) break;
		if (i >= numEdges) break;
	} while(1);
	fprintf( f, "\n");

	// Output faces
	i=0;
	currFace = s->sfaces;
	do 
	{
		Loop *floop = currFace->floop;
		HalfEdge *e1 = floop->ledges;
		HalfEdge *e2 = e1->next;
		HalfEdge *e3 = e2->next;
		fprintf(f, "t %d %d %d %d %d %d %d\n", 
				currFace->faceno,
				e1->hvert->vertexno, e2->hvert->vertexno, e3->hvert->vertexno,
				e1->hedge->edgeno, e2->hedge->edgeno, e3->hedge->edgeno );
		i++;
		currFace = currFace->next;
		if (!currFace || currFace == s->sfaces) break;
		if (i >= numTris) break;
	} while(1);
	fprintf( f, "\n");


	fclose( f );
	return 0;
}

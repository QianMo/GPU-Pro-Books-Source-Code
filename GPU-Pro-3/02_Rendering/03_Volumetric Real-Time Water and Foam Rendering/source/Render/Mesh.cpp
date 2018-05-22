#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <GL/gl.h>

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "../Render/Mesh.h"
#include "../Graphics/OBJLoader.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- Mesh::Mesh ------------------------------------------
// -----------------------------------------------------------------------------
Mesh::Mesh(void) : RenderObject()
{
}

// -----------------------------------------------------------------------------
// ----------------------- Mesh::~Mesh -----------------------------------------
// -----------------------------------------------------------------------------
Mesh::~Mesh(void)
{
}


// -----------------------------------------------------------------------------
// ----------------------- Mesh::Init ------------------------------------------
// -----------------------------------------------------------------------------
void Mesh::Init(const int& idNum, const char* fileName, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	assert((addToPhysicWorld!=true) || (isLevelElement!=true));

	id = idNum;
	materialId = matId;
	objectType = RenderObject::TYPE_MESH;

	int len = static_cast<int>(strlen(fileName));

	if ((fileName[len-3] == 'o' || fileName[len-3] == 'O') &&
		(fileName[len-2] == 'b' || fileName[len-2] == 'B') &&
		(fileName[len-1] == 'j' || fileName[len-1] == 'J'))
	{
		OBJLoader wo;
		wo.LoadObj(fileName, true);

		assert(wo.mVertexCount != 0);

		numVertices = wo.mVertexCount;
		numIndices = wo.mTriCount*3;

		// allocate space for the vertices and indices
		vertices = new RenderObject::Vertex[numVertices];
		indices = new unsigned[numIndices];

		// copy positions and indices
		Vector3* vSrcPos = (Vector3*)wo.mVertices;
		Vector3* vSrcNor = (Vector3*)wo.mNormals;
		Vector2* vSrcTex = (Vector2*)wo.mTexCoords;

		int i;
		for (i=0; i<numVertices; i++, vSrcPos++, vSrcNor++, vSrcTex++)
		{
			vertices[i].vertex.x = (*vSrcPos).x;
			vertices[i].vertex.y = (*vSrcPos).y;
			vertices[i].vertex.z = (*vSrcPos).z;

			vertices[i].normal.x = (*vSrcNor).x;
			vertices[i].normal.y = (*vSrcNor).y;
			vertices[i].normal.z = (*vSrcNor).z;

			vertices[i].texture.u = (*vSrcTex).u;
			vertices[i].texture.v = (*vSrcTex).v;
		}

		memcpy((unsigned int*)indices, wo.mIndices, sizeof(unsigned int)*numIndices);

		wo.Exit();
	}
	else
	{
		assert(false);
	}

	RenderObject::Init();
}

// -----------------------------------------------------------------------------
// -------------------------- Mesh::InitCollisionMesh --------------------------
// -----------------------------------------------------------------------------
void Mesh::InitCollisionMesh(const char* fileName)
{
	id = 0;
	materialId = 0;
	objectType = RenderObject::TYPE_MESH;

	int len = static_cast<int>(strlen(fileName));

	if ((fileName[len-3] == 'o' || fileName[len-3] == 'O') &&
		(fileName[len-2] == 'b' || fileName[len-2] == 'B') &&
		(fileName[len-1] == 'j' || fileName[len-1] == 'J'))
	{
		OBJLoader wo;
		wo.LoadObj(fileName, true);

		assert(wo.mVertexCount != 0);

		numVertices = wo.mVertexCount;
		numIndices = wo.mTriCount*3;

		// allocate space for the vertices and indices
		vertices = new RenderObject::Vertex[numVertices];
		indices = new unsigned[numIndices];

		// copy positions and indices
		Vector3* vSrcPos = (Vector3*)wo.mVertices;
		Vector3* vSrcNor = (Vector3*)wo.mNormals;
		Vector2* vSrcTex = (Vector2*)wo.mTexCoords;

		int i;
		for (i=0; i<numVertices; i++, vSrcPos++, vSrcNor++, vSrcTex++)
		{
			vertices[i].vertex.x = (*vSrcPos).x;
			vertices[i].vertex.y = (*vSrcPos).y;
			vertices[i].vertex.z = (*vSrcPos).z;

			vertices[i].normal.x = (*vSrcNor).x;
			vertices[i].normal.y = (*vSrcNor).y;
			vertices[i].normal.z = (*vSrcNor).z;

			vertices[i].texture.u = (*vSrcTex).u;
			vertices[i].texture.v = (*vSrcTex).v;
		}

		memcpy((unsigned int*)indices, wo.mIndices, sizeof(unsigned int)*numIndices);

		wo.Exit();
	}
	else
	{
		assert(false);
	}
}

// -----------------------------------------------------------------------------
// ----------------------- Mesh::SetVertexData ---------------------------------
// -----------------------------------------------------------------------------
void Mesh::SetVertexData(const int& index,
						   const float& tu, const float& tv, 
						   const float& nx, const float& ny, const float& nz, 
						   const float& vx, const float& vy, const float& vz)	
{
	(vertices+index)->texture.u = tu;
	(vertices+index)->texture.v = tv;
	(vertices+index)->normal.x = nx;
	(vertices+index)->normal.y = ny;
	(vertices+index)->normal.z = nz;
	(vertices+index)->vertex.x = vx;
	(vertices+index)->vertex.y = vy;
	(vertices+index)->vertex.z = vz;
}
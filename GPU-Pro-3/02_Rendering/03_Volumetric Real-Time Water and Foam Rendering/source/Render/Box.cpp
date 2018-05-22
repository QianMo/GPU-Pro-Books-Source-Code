#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <GL/gl.h>

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "../Render/Box.h"
#include "../Util/Vector3.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- Box::Box --------------------------------------------
// -----------------------------------------------------------------------------
Box::Box(void) : RenderObject()
{
}

// -----------------------------------------------------------------------------
// ----------------------- Box::~Box -------------------------------------------
// -----------------------------------------------------------------------------
Box::~Box(void)
{
	Exit();
}


// -----------------------------------------------------------------------------
// ----------------------- Box::Init -------------------------------------------
// -----------------------------------------------------------------------------
void Box::Init(const int& idNum, const Vector3& pos, const float& w, const float& h, const float& d,
			   const Vector3& su, const Vector3& sv, const int& matId, const bool& addToPhysicWorld,
			   const bool& isLevelElement)
{
	assert((addToPhysicWorld!=true) || (isLevelElement!=true));

	numVertices = 24;

	id = idNum;
	width = w;
	height = h;
	depth = d;
	materialId = matId;
	objectType = RenderObject::TYPE_QUADER;

	if( vertices != NULL )
	{
		delete []vertices;
		vertices = NULL;
		vertices = new Vertex[numVertices];
	}
	else
	{
		vertices = new Vertex[numVertices];
	}

	ZeroMemory(vertices, numVertices*sizeof(Vertex));

	Vector3 suT;
	Vector3 svT;

	if (su.y == 0 && su.z == 0 && sv.x == 0 && sv.y == 0 && sv.z == 0)
	{
		suT.x = depth / su.x;
		suT.y = depth / su.x;
		suT.z = width / su.x;

		svT.x = height / su.x;
		svT.y = width / su.x;
		svT.z = height / su.x;
	}
	else
	{
		suT = su;
		svT = sv;
	}

	SetVertexData(0, suT.y, svT.y, /**/ 0.0f, 0.0f, 1.0f, /**/ 1.0f, 0.0f, 0.0f, /**/ 0.0f, 1.0f, 0.0f, pos.x + w / 2.0f, pos.y + h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(1, 0.0f, svT.y, /**/ 0.0f, 0.0f, 1.0f, /**/ 1.0f, 0.0f, 0.0f, /**/0.0f, 1.0f, 0.0f, pos.x + w / 2.0f, pos.y + h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(2, suT.y, 0.0f, /**/ 0.0f, 0.0f, 1.0f, /**/ 1.0f, 0.0f, 0.0f, /**/0.0f, 1.0f, 0.0f, pos.x - w / 2.0f, pos.y + h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(3, 0.0f, 0.0f, /**/ 0.0f, 0.0f, 1.0f, /**/ 1.0f, 0.0f, 0.0f, /**/0.0f, 1.0f, 0.0f, pos.x - w / 2.0f, pos.y + h / 2.0f, pos.z - d / 2.0f);

	SetVertexData(4, 0.0f, 0.0f, /**/ -1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, -1.0f, pos.x + w / 2.0f, pos.y + h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(5, 0.0f, svT.z, /**/ -1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, -1.0f, pos.x + w / 2.0f, pos.y - h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(6, suT.z, 0.0f, /**/ -1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, -1.0f, pos.x - w / 2.0f, pos.y + h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(7, suT.z, svT.z, /**/ -1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, -1.0f, pos.x - w / 2.0f, pos.y - h / 2.0f, pos.z - d / 2.0f);

	SetVertexData(8, 0.0f, 0.0f, /**/ 0.0f, 0.0f, -1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/1.0f, 0.0f, 0.0f, pos.x + w / 2.0f, pos.y + h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(9, 0.0f, svT.x, /**/ 0.0f, 0.0f, -1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/1.0f, 0.0f, 0.0f, pos.x + w / 2.0f, pos.y - h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(10, suT.x, 0.0f, /**/ 0.0f, 0.0f, -1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/1.0f, 0.0f, 0.0f, pos.x + w / 2.0f, pos.y + h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(11, suT.x, svT.x, /**/ 0.0f, 0.0f, -1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/1.0f, 0.0f, 0.0f, pos.x + w / 2.0f, pos.y - h / 2.0f, pos.z - d / 2.0f);

	SetVertexData(12, 0.0f, 0.0f, /**/ 1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, 1.0f, pos.x - w / 2.0f, pos.y + h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(13, 0.0f, svT.z, /**/ 1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, 1.0f, pos.x - w / 2.0f, pos.y - h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(14, suT.z, 0.0f, /**/ 1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, 1.0f, pos.x + w / 2.0f, pos.y + h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(15, suT.z, svT.z, /**/ 1.0f, 0.0f, 0.0f, /**/ 0.0f, -1.0f, 0.0f, /**/0.0f, 0.0f, 1.0f, pos.x + w / 2.0f, pos.y - h / 2.0f, pos.z + d / 2.0f);

	SetVertexData(16, 0.0f, 0.0f, /**/ 0.0f, 0.0f, 1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/-1.0f, 0.0f, 0.0f, pos.x - w / 2.0f, pos.y + h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(17, 0.0f, svT.x, /**/ 0.0f, 0.0f, 1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/-1.0f, 0.0f, 0.0f, pos.x - w / 2.0f, pos.y - h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(18, suT.x, 0.0f, /**/ 0.0f, 0.0f, 1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/-1.0f, 0.0f, 0.0f, pos.x - w / 2.0f, pos.y + h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(19, suT.x, svT.x, /**/ 0.0f, 0.0f, 1.0f, /**/ 0.0f, -1.0f, 0.0f, /**/-1.0f, 0.0f, 0.0f, pos.x - w / 2.0f, pos.y - h / 2.0f, pos.z + d / 2.0f);

	SetVertexData(20, 0.0f, svT.y, /**/ 0.0f, 0.0f, -1.0f, /**/ -1.0f, 0.0f, 0.0f, /**/0.0f, -1.0f, 0.0f, pos.x - w / 2.0f, pos.y - h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(21, suT.y, svT.y, /**/ 0.0f, 0.0f, -1.0f, /**/ -1.0f, 0.0f, 0.0f, /**/0.0f, -1.0f, 0.0f, pos.x - w / 2.0f, pos.y - h / 2.0f, pos.z - d / 2.0f);
	SetVertexData(22, 0.0f, 0.0f, /**/ 0.0f, 0.0f, -1.0f, /**/ -1.0f, 0.0f, 0.0f, /**/0.0f, -1.0f, 0.0f, pos.x + w / 2.0f, pos.y - h / 2.0f, pos.z + d / 2.0f);
	SetVertexData(23, suT.y, 0.0f, /**/ 0.0f, 0.0f, -1.0f, /**/ -1.0f, 0.0f, 0.0f, /**/0.0f, -1.0f, 0.0f, pos.x + w / 2.0f, pos.y - h / 2.0f, pos.z - d / 2.0f);

	RenderObject::Init();
}


// -----------------------------------------------------------------------------
// ----------------------- Box::SetVertexData ----------------------------------
// -----------------------------------------------------------------------------
void Box::SetVertexData(const int& index,
						const float& tu, const float& tv,
						const float& tanx, const float& tany, const float& tanz,
						const float& binx, const float& biny, const float& binz, 
						const float& nx, const float& ny, const float& nz, 
						const float& vx, const float& vy, const float& vz)	
{
	(vertices+index)->texture.u = tu;
	(vertices+index)->texture.v = tv;

	(vertices+index)->tangent.x = tanx;
	(vertices+index)->tangent.y = tany;
	(vertices+index)->tangent.z = tanz;
	(vertices+index)->binormal.x = binx;
	(vertices+index)->binormal.y = biny;
	(vertices+index)->binormal.z = binz;
	(vertices+index)->binormalHandedness = 1.0f;

	(vertices+index)->normal.x = nx;
	(vertices+index)->normal.y = ny;
	(vertices+index)->normal.z = nz;
	(vertices+index)->vertex.x = vx;
	(vertices+index)->vertex.y = vy;
	(vertices+index)->vertex.z = vz;
}
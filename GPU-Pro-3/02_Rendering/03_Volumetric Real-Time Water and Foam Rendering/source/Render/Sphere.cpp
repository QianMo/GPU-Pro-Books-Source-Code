#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <GL/gl.h>

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "../Render/Sphere.h"
#include "../Util/Vector3.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- Sphere::Sphere --------------------------------------
// -----------------------------------------------------------------------------
Sphere::Sphere(void) : RenderObject()
{
}

// -----------------------------------------------------------------------------
// ----------------------- Sphere::~Sphere -------------------------------------
// -----------------------------------------------------------------------------
Sphere::~Sphere(void)
{
	Exit();
}


// -----------------------------------------------------------------------------
// ----------------------- Sphere::Init ----------------------------------------
// -----------------------------------------------------------------------------
void Sphere::Init(const int& idNum, const float& r, const int& p, const Vector3& pos, const int& matId, const bool& addToPhysicWorld, const bool& isLevelElement)
{
	assert((addToPhysicWorld!=true) || (isLevelElement!=true));

	id = idNum;
	radius = r;
	precision = p;
	materialId = matId;
	objectType = RenderObject::TYPE_SPHERE;

	if (isLevelElement)
		CreateSphereGeometry(pos.x, pos.y, pos.z, radius, precision);
	else
		CreateSphereGeometry(0.0f, 0.0f, 0.0f, radius, precision);
	RenderObject::Init();
}


// -----------------------------------------------------------------------------
// ----------------------- Sphere::CreateSphereGeometry ------------------------
// -----------------------------------------------------------------------------
void Sphere::CreateSphereGeometry(const float& cx, const float& cy, const float& cz, float& r, int& p)	
{
	//const float PI = 3.14159265358979f;
	const float TWOPI = 6.28318530717958f;
	const float PIDIV2 = 1.57079632679489f;

	float theta1 = 0.0;
	float theta2 = 0.0;
	float theta3 = 0.0;

	float ex = 0.0f;
	float ey = 0.0f;
	float ez = 0.0f;

	float px = 0.0f;
	float py = 0.0f;
	float pz = 0.0f;

	float tu  = 0.0f;
	float tv  = 0.0f;

	numVertices = (p/2) * ((p+1)*2);

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

	// disallow a negative number for radius.
	if( r < 0 )
		r = -r;

	// disallow a negative number for precision.
	if( p < 4 ) 
		p = 4;

	int k = -1;

	Vector3 tangent;
	Vector3 binormal;

	for( int i = 0; i < p/2; ++i )
	{
		theta1 = i * TWOPI / p - PIDIV2;
		theta2 = (i + 1) * TWOPI / p - PIDIV2;

		for( int j = 0; j <= p; ++j )
		{
			theta3 = j * TWOPI / p;

			ex = cosf(theta1) * cosf(theta3);
			ey = sinf(theta1);
			ez = cosf(theta1) * sinf(theta3);
			px = cx + r * ex;
			py = cy + r * ey;
			pz = cz + r * ez;
			tu  = -(j/(float)p);
			tv  = 2*i/(float)p;

			tangent = Vector3(ez, 0.0f, -ex);

			tangent.Normalize();
			binormal = (Vector3(ex, ey, ez)).CrossProduct(tangent);

			++k;
			SetVertexData( k, tu, tv, tangent.x, tangent.y, tangent.z, binormal.x, binormal.y, binormal.z, ex, ey, ez, px, py, pz );

			ex = cosf(theta2) * cosf(theta3);
			ey = sinf(theta2);
			ez = cosf(theta2) * sinf(theta3);
			px = cx + r * ex;
			py = cy + r * ey;
			pz = cz + r * ez;
			tu  = -(j/(float)p);
			tv  = 2*(i+1)/(float)p;

			tangent = Vector3(ez, 0.0f, -ex);

			tangent.Normalize();
			binormal = (Vector3(ex, ey, ez)).CrossProduct(tangent);

			++k;
			SetVertexData( k, tu, tv, tangent.x, tangent.y, tangent.z, binormal.x, binormal.y, binormal.z, ex, ey, ez, px, py, pz );
		}
	}
}


// -----------------------------------------------------------------------------
// ----------------------- Sphere::SetVertexData -------------------------------
// -----------------------------------------------------------------------------
void Sphere::SetVertexData(const int& index,
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
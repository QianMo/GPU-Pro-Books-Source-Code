#include <assert.h>
#include <stdio.h>

#include "../Util/BoundingSphere.h"

// -----------------------------------------------------------------------------
// -------------------------------- BoundingSphere::BoundingSphere -------------
// -----------------------------------------------------------------------------
BoundingSphere::BoundingSphere(void)
{
	Reset();
}

// -----------------------------------------------------------------------------
// ------------------------------- BoundingSphere::~BoundingSphere -------------
// -----------------------------------------------------------------------------
BoundingSphere::~BoundingSphere(void)
{
}


// -----------------------------------------------------------------------------
// ------------------------------- BoundingSphere::ResetBoundingSphere ---------
// -----------------------------------------------------------------------------
void BoundingSphere::Reset(void)
{
	position = Vector3(0, 0, 0);
	radius = 0;
}

// -----------------------------------------------------------------------------
// ------------------------------- BoundingSphere::AddVertices -----------------
// -----------------------------------------------------------------------------
void BoundingSphere::AddVertices(const Vertex* vertices, int numVertices)
{
	assert(vertices != NULL);
	assert(numVertices > 0);
	Reset();

	int i;
	for (i=0; i<numVertices; i++)
	{
		position += vertices[i].vertex;
	}
	position /= (float)numVertices;

	// find farthest point in set
	for (i=0; i<numVertices; i++)
	{
		Vector3 v;
		v = vertices[i].vertex - position;
		float distSq = v.SquaredLength();
		if (distSq>radius)
			radius = distSq;
	}

	radius = sqrtf(radius);
}

// -----------------------------------------------------------------------------
// ------------------ BoundingSphere::IntersectsSphere -------------------------
// -----------------------------------------------------------------------------
bool BoundingSphere::Intersects(const BoundingSphere& otherSphere) const
{
	Vector3 v;

	v = position-otherSphere.GetPosition();
	float centDist = radius+otherSphere.GetRadius();

	return (v.SquaredLength() <= centDist*centDist);
}


// -----------------------------------------------------------------------------
// ------------------ BoundingSphere::IntersectsRay ----------------------------
// -----------------------------------------------------------------------------
bool BoundingSphere::Intersects(const Ray& ray, float intersectEnd) const
{
	Vector3 dst = ray.GetOrigin() - position;
	float B = dst.DotProduct(ray.GetDirection());
	float C = dst.DotProduct(dst) - radius*radius;
	float D = B*B - C;
	if (D < 0)
		return false;

	float sqrtD = sqrt(D);
	float t0 = (-B - sqrtD);
	//float t1 = (-B + sqrtD); // the second intersection

	if (t0 < intersectEnd)
		return true;
	else
		return false;
}

// -----------------------------------------------------------------------------
// ------------------------------- BoundingSphere::IsInside --------------------
// -----------------------------------------------------------------------------
bool BoundingSphere::Intersects(const Math::FrustumPlane* viewFrustum, const Matrix4& objectMatrix) const
{
	Vector3 transformedPosition = position;
	transformedPosition = objectMatrix * transformedPosition;

	for( int i = 0; i < 6; ++i )
	{
		if( viewFrustum[i].plane[0] * transformedPosition.x +
			viewFrustum[i].plane[1] * transformedPosition.y +
			viewFrustum[i].plane[2] * transformedPosition.z +
			viewFrustum[i].plane[3] <= -radius )
			return false;
	}

	return true;
}

// -----------------------------------------------------------------------------
// ------------------------------- BoundingSphere::IsInside --------------------
// -----------------------------------------------------------------------------
bool BoundingSphere::IsInside(const Vector3& point) const
{
	Vector3 v;

	v=position-point;

	if (v.SquaredLength() > (radius*radius))
		return false;

	return true;
}
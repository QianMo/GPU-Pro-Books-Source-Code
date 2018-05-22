#include <assert.h>

#include "AxisAlignedBoundingBox.h"
#include "../Util/Ray.h"

// -----------------------------------------------------------------------------
// -------------- AxisAlignedBoundingBox::AxisAlignedBoundingBox ---------------
// -----------------------------------------------------------------------------
AxisAlignedBoundingBox::AxisAlignedBoundingBox(void)
{
	Reset();
}

// -----------------------------------------------------------------------------
// ------------- AxisAlignedBoundingBox::~AxisAlignedBoundingBox ---------------
// -----------------------------------------------------------------------------
AxisAlignedBoundingBox::~AxisAlignedBoundingBox(void)
{
	Destroy();
}

// -----------------------------------------------------------------------------
// --------------- AxisAlignedBoundingBox::Destroy -----------------------------
// -----------------------------------------------------------------------------
void AxisAlignedBoundingBox::Destroy(void)
{
}

// -----------------------------------------------------------------------------
// ------------------ AxisAlignedBoundingBox::Reset ----------------------------
// -----------------------------------------------------------------------------
void AxisAlignedBoundingBox::Reset(void)
{
	minVertex = Vector3(10000.0f, 10000.0f, 10000.0f);
	maxVertex = Vector3(-10000.0f, -10000.0f, -10000.0f);
}

// -----------------------------------------------------------------------------
// -------------- AxisAlignedBoundingBox::AddVertex ----------------------------
// -----------------------------------------------------------------------------
void AxisAlignedBoundingBox::AddVertex(const Vector3& vertex)
{
	if (vertex.x < minVertex.x)
		minVertex.x = vertex.x;
	if (vertex.x > maxVertex.x)
		maxVertex.x = vertex.x;

	if (vertex.y < minVertex.y)
		minVertex.y = vertex.y;
	if (vertex.y > maxVertex.y)
		maxVertex.y = vertex.y;

	if (vertex.z < minVertex.z)
		minVertex.z = vertex.z;
	if (vertex.z > maxVertex.z)
		maxVertex.z = vertex.z;
}

// -----------------------------------------------------------------------------
// ---------------- AxisAlignedBoundingBox::AddVertex --------------------------
// -----------------------------------------------------------------------------
void AxisAlignedBoundingBox::AddVertices(const Vertex* vertices, unsigned int numVertices)
{
	for (unsigned int i=0; i<numVertices; i++)
	{
		if (vertices[i].vertex.x < minVertex.x)
			minVertex.x = vertices[i].vertex.x;
		if (vertices[i].vertex.x > maxVertex.x)
			maxVertex.x = vertices[i].vertex.x;

		if (vertices[i].vertex.y < minVertex.y)
			minVertex.y = vertices[i].vertex.y;
		if (vertices[i].vertex.y > maxVertex.y)
			maxVertex.y = vertices[i].vertex.y;

		if (vertices[i].vertex.z < minVertex.z)
			minVertex.z = vertices[i].vertex.z;
		if (vertices[i].vertex.z > maxVertex.z)
			maxVertex.z = vertices[i].vertex.z;
	}
}

// -----------------------------------------------------------------------------
// ----------- AxisAlignedBoundingBox::AddAxisAlignedBoundingBox ---------------
// -----------------------------------------------------------------------------
void AxisAlignedBoundingBox::AddAxisAlignedBoundingBox(const AxisAlignedBoundingBox& aabb)
{
	AddVertex(aabb.GetMin());
	AddVertex(aabb.GetMax());
}

// -----------------------------------------------------------------------------
// ------------------- AxisAlignedBoundingBox::GetPosition ---------------------
// -----------------------------------------------------------------------------
Vector3 AxisAlignedBoundingBox::GetCenter(void) const
{
	return (minVertex + maxVertex) / 2;
}


// -----------------------------------------------------------------------------
// ----------------- AxisAlignedBoundingBox::GetDimension ----------------------
// -----------------------------------------------------------------------------
Vector3 AxisAlignedBoundingBox::GetDimension(void) const
{
	return Vector3(maxVertex.x - minVertex.x, maxVertex.y - minVertex.y, maxVertex.z - minVertex.z);
}

// -----------------------------------------------------------------------------
// -------------- AxisAlignedBoundingBox::BoxCollisionTest ---------------------
// -----------------------------------------------------------------------------
bool AxisAlignedBoundingBox::Intersects(const AxisAlignedBoundingBox& otherBox) const
{
	const Vector3& minOther = otherBox.GetMin();
	const Vector3& maxOther = otherBox.GetMax();

	if ((minVertex.x > maxOther.x) || (minOther.x > maxVertex.x))
		return false;

	if ((minVertex.y > maxOther.y) || (minOther.y > maxVertex.y))
		return false;

	if ((minVertex.z > maxOther.z) || (minOther.z > maxVertex.z))
		return false;

	return true;
}

// -----------------------------------------------------------------------------
// ---------------- AxisAlignedBoundingBox::BoxRayCollisionTest ----------------
// -----------------------------------------------------------------------------
bool AxisAlignedBoundingBox::Intersects(const Ray& ray, float intersectEnd) const
{
	Vector3 rayDelta = ray.GetDirection() * intersectEnd;
	bool inside = true;
	float xt = 0.0f;
	float yt = 0.0f;
	float zt = 0.0f;

	Vector3 rayPosition = ray.GetOrigin();

	if(rayPosition.x < minVertex.x)
	{
		xt = minVertex.x - rayPosition.x;

		if(xt > rayDelta.x) // If the ray is moving away from the AxisAlignedBoundingBox, there is no intersection 
			return false;

		xt /= rayDelta.x; 
		inside = false;
	} 
	else if(rayPosition.x > maxVertex.x)
	{
		xt = maxVertex.x - rayPosition.x;

		if(xt < rayDelta.x) // If the ray is moving away from the AxisAlignedBoundingBox, there is no intersection 
			return false;

		xt /= rayDelta.x;
		inside = false;
	} 
	else
	{
		xt = -1.0f; 
	}

	// Test the X component of the ray's origin to see if we are inside or not
	if(rayPosition.y < minVertex.y)
	{
		yt = minVertex.y - rayPosition.y;

		if(yt > rayDelta.y) // If the ray is moving away from the AxisAlignedBoundingBox, there is no intersection 
			return false;

		yt /= rayDelta.y;
		inside = false;
	} 
	else if(rayPosition.y > maxVertex.y)
	{
		yt = maxVertex.y - rayPosition.y;

		if(yt < rayDelta.y) // If the ray is moving away from the AxisAlignedBoundingBox, there is no intersection 
			return false;

		yt /= rayDelta.y;
		inside = false;
	} 
	else
	{
		yt = -1.0f;
	}

	if(rayPosition.z < minVertex.z)
	{
		zt = minVertex.z - rayPosition.z;

		if(zt > rayDelta.z) // If the ray is moving away from the AxisAlignedBoundingBox, there is no intersection 
			return false;

		zt /= rayDelta.z;
		inside = false;
	} 
	else if(rayPosition.z > maxVertex.z)
	{
		zt = maxVertex.z - rayPosition.z;

		if(zt < rayDelta.z) // If the ray is moving away from the AxisAlignedBoundingBox, there is no intersection 
			return false;

		zt /= rayDelta.z;
		inside = false;
	} 
	else
	{
		zt = -1.0f;
	}

	// If the origin inside the AxisAlignedBoundingBox
	if(inside)
		return true; // The ray intersects the AxisAlignedBoundingBox

	// We want to test the AxisAlignedBoundingBox planes with largest value out of xt, yt, and zt.  So
	// first we determine which value is the largest.

	if((xt >= yt) && (xt >= zt)) // If the ray intersects with the AxisAlignedBoundingBox's YZ plane
	{
		// Compute intersection values
		float y = rayPosition.y + rayDelta.y * xt;
		float z = rayPosition.z + rayDelta.z * xt;

		// Test to see if collision takes place within the bounds of the AxisAlignedBoundingBox
		if(y < minVertex.y || y > maxVertex.y)
			return false;
		else if(z < minVertex.z || z > maxVertex.z)
			return false;
	}
	else if((yt >= xt) &&(yt >= zt)) // Intersects with the XZ plane
	{
		// Compute intersection values
		float x = rayPosition.x + rayDelta.x * yt;
		float z = rayPosition.z + rayDelta.z * yt;

		// Test to see if collision takes place within the bounds of the AxisAlignedBoundingBox
		if(x < minVertex.x || x > maxVertex.x)
			return false;
		else if(z < minVertex.z || z > maxVertex.z) 
			return false;
	}
	else // Intersects with the XY plane
	{
		// Compute intersection values
		float x = rayPosition.x + rayDelta.x * zt;
		float y = rayPosition.y + rayDelta.y * zt;

		// Test to see if collision takes place within the bounds of the AxisAlignedBoundingBox
		if(x < minVertex.x || x > maxVertex.x)
			return false;
		else if(y < minVertex.y || y > maxVertex.y)
			return false;
	}

	// The ray intersects the AxisAlignedBoundingBox
	return true;
}

// -----------------------------------------------------------------------------
// -------------- AxisAlignedBoundingBox::IsInside -----------------------------
// -----------------------------------------------------------------------------
bool AxisAlignedBoundingBox::IsInside(const Vector3& point) const
{
	if ((point.x<minVertex.x) || (point.x>maxVertex.x))
		return false;
	if ((point.y<minVertex.y) || (point.y>maxVertex.y))
		return false;
	if ((point.z<minVertex.z) || (point.z>maxVertex.z))
		return false;

	return true;
}
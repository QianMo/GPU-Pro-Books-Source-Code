#ifndef __AXISALIGNEDBOUNDINGBOX_H__
#define __AXISALIGNEDBOUNDINGBOX_H__

#include "../Util/Vector3.h"
#include "../Util/Vector2.h"
#include "../Util/Ray.h"

class Ray;


// -----------------------------------------------------------------------------
/// AxisAlignedBoundingBox
// -----------------------------------------------------------------------------
/// 
/// 
// -----------------------------------------------------------------------------
class AxisAlignedBoundingBox
{
public:
	struct Vertex
	{
		Vector2 texture;
		Vector3 tangent;
		Vector3 binormal;
		float binormalHandedness;
		Vector3 normal;
		Vector3 vertex;
	};

	AxisAlignedBoundingBox(void);
	~AxisAlignedBoundingBox(void);

	void Destroy(void);
	void AddVertex(const Vector3& vertex);
	void AddVertices(const Vertex* vertices, unsigned int numVertices);
	void AddAxisAlignedBoundingBox(const AxisAlignedBoundingBox& aabb);
	void Reset(void);
	Vector3 GetCenter(void) const;
	Vector3 GetDimension(void) const;
	bool Intersects(const AxisAlignedBoundingBox& otherBox) const;
	bool Intersects(const Ray& ray, float intersectEnd) const;
	bool IsInside(const Vector3& point) const;
	const Vector3& GetMin(void) const { return minVertex; }
	const Vector3& GetMax(void) const { return maxVertex; }
	void SetMin(const Vector3& minToSet) { minVertex = minToSet; }
	void SetMax(const Vector3& maxToSet) { maxVertex = maxToSet; }

private:
	Vector3 minVertex;
	Vector3 maxVertex;
};

#endif

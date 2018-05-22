#ifndef __BOUNDINGSPHERE_H__
#define __BOUNDINGSPHERE_H__

#include "../Util/Vector3.h"
#include "../Util/Vector2.h"
#include "../Util/Math.h"
#include "../Util/Ray.h"


// -----------------------------------------------------------------------------
/// BoundingSphere
// ----------------------------------------------------------------------------- 
/// 
/// 
// -----------------------------------------------------------------------------
class BoundingSphere
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

	BoundingSphere(void);
	~BoundingSphere(void);

	void AddVertices(const Vertex* vertices, int numVertices);
	void Reset(void);
	bool Intersects(const BoundingSphere& otherSphere) const;
	bool Intersects(const Ray& ray, float intersectEnd) const;
	bool Intersects(const Math::FrustumPlane* viewFrustum, const Matrix4& objectMatrix) const;
	bool IsInside(const Vector3& point) const;
	const Vector3& GetPosition(void) const { return position; }
	float GetRadius(void) const { return radius; }

private:
	Vector3 position;
	float radius;
};

#endif //__BOUNDINGSPHERE_H__

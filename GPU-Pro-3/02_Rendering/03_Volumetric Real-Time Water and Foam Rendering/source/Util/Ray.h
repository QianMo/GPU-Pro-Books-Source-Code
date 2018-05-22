#ifndef __RAY_H__
#define __RAY_H__

#include "../Util/Vector3.h"
#include "../Util/Matrix4.h"

// -----------------------------------------------------------------------------
/// Ray
// ----------------------------------------------------------------------------- 
/// 
/// 
// -----------------------------------------------------------------------------
class Ray
{
public:
	Ray(void);
	Ray(const Vector3& pos, const Vector3& dir);
	~Ray(void);

	const Vector3& GetOrigin(void) const { return origin; }
	const Vector3& GetDirection(void) const { return direction; }

	void Transform(const Matrix4& matrix);

private:
	Vector3 origin;
	Vector3 direction;
};

#endif

#include "../Util/Ray.h"
#include "../Util/Matrix4.h"

// -----------------------------------------------------------------------------
// -------------------------------- Ray::Ray -----------------------------------
// -----------------------------------------------------------------------------
Ray::Ray(void)
{
	origin = Vector3(0, 0, 0);
	direction = Vector3(0, 1, 0);
}

// -----------------------------------------------------------------------------
// -------------------------------- Ray::Ray -----------------------------------
// -----------------------------------------------------------------------------
Ray::Ray(const Vector3& pos, const Vector3& dir)
{
	origin = pos;
	direction = dir;

	direction.Normalize();
}

// -----------------------------------------------------------------------------
// ------------------------------- Ray::~Ray -----------------------------------
// -----------------------------------------------------------------------------
Ray::~Ray(void)
{
}

// -----------------------------------------------------------------------------
// ------------------------------- Ray::Transform --------------------------------
// -----------------------------------------------------------------------------
void Ray::Transform(const Matrix4& matrix)
{
	origin = matrix * origin;

	Vector3 newDirection;
		newDirection.x =
			matrix.entry[0]*direction.x +
			matrix.entry[4]*direction.y +
			matrix.entry[8]*direction.z;
		newDirection.y =
			matrix.entry[1]*direction.x +
			matrix.entry[5]*direction.y +
			matrix.entry[9]*direction.z;
		newDirection.z =
			matrix.entry[2]*direction.x +
			matrix.entry[6]*direction.y +
			matrix.entry[10]*direction.z;

	direction = newDirection;
}
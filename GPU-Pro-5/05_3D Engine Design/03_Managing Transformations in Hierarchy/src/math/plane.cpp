#include <math/plane.hpp>
#include <math/matrix.hpp>



void Plane::Transform(Matrix transform)
{
	Vector4 v(a, b, c, d);
	transform.Inverse();
	transform.Transpose();

	v *= transform;

	a = v.x;
	b = v.y;
	c = v.z;
	d = v.w;

	Normalize();
}

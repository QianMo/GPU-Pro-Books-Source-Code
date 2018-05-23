#include <math/vector.hpp>
#include <math/matrix.hpp>



// ----------------------------------------------------------------------------



Vector2::Vector2(const Vector3& v)
{
	x = v.x;
	y = v.y;
}



Vector2::Vector2(const Vector4& v)
{
	x = v.x;
	y = v.y;
}



// ----------------------------------------------------------------------------



const Vector3 Vector3::AxisX = Vector3(1.0f, 0.0f, 0.0f);
const Vector3 Vector3::AxisY = Vector3(0.0f, 1.0f, 0.0f);
const Vector3 Vector3::AxisZ = Vector3(0.0f, 0.0f, 1.0f);
const Vector3 Vector3::Zero = Vector3(0.0f);
const Vector3 Vector3::One = Vector3(1.0f);



Vector3::Vector3(const Vector2& v)
{
	x = v.x;
	y = v.y;
	z = 0.0f;
}



Vector3::Vector3(const Vector4& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
}



Vector3 Vector3::operator * (const Matrix& m) const
{
	Vector3 temp;

	temp.x = x*m(0, 0) + y*m(1, 0) + z*m(2, 0);
	temp.y = x*m(0, 1) + y*m(1, 1) + z*m(2, 1);
	temp.z = x*m(0, 2) + y*m(1, 2) + z*m(2, 2);

	return temp;
}



Vector3& Vector3::operator *= (const Matrix& m)
{
	*this = *this * m;
	return *this;
}



// ----------------------------------------------------------------------------



Vector4::Vector4(const Vector2& v)
{
	x = v.x;
	y = v.y;
	z = 0.0f;
	w = 1.0f;
}



Vector4::Vector4(const Vector3& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	w = 1.0f;
}



Vector4 Vector4::operator * (const Matrix& m) const
{
	Vector4 temp;

	temp.x = x*m(0, 0) + y*m(1, 0) + z*m(2, 0) + w*m(3, 0);
	temp.y = x*m(0, 1) + y*m(1, 1) + z*m(2, 1) + w*m(3, 1);
	temp.z = x*m(0, 2) + y*m(1, 2) + z*m(2, 2) + w*m(3, 2);
	temp.w = x*m(0, 3) + y*m(1, 3) + z*m(2, 3) + w*m(3, 3);

	return temp;
}



Vector4& Vector4::operator *= (const Matrix& m)
{
	*this = *this * m;
	return *this;
}

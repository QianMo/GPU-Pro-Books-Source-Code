#ifndef __VECTOR3_H__
#define __VECTOR3_H__


// -----------------------------------------------------------------------------
/// Vector3
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Vector3
{
public:
	Vector3() : x(0), y(0), z(0) {};
	Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) { };
	Vector3(const Vector3& other) : x(other.x), y(other.y), z(other.z) { };
	const Vector3& operator=(const Vector3& other);

	bool operator!=(const Vector3& other) const { return ((x != other.x) || (y != other.y) || (z != other.z)); }
	bool operator==(const Vector3& other) const { return ((x == other.x) && (y == other.y) && (z == other.z)); }
	bool operator<(const Vector3& other) const	{ return x < other.x; }

	Vector3 operator-() const { return Vector3(-x, -y, -z); }

	Vector3 operator+(const Vector3& other) const { return Vector3(x + other.x, y + other.y, z + other.z); }
	const Vector3& operator+=(const Vector3& other) { x += other.x; y += other.y; z += other.z; return (*this); }    

	Vector3 operator-(const Vector3& other) const { return Vector3(x - other.x, y - other.y, z - other.z); }
	const Vector3& operator-=(const Vector3& other) { x -= other.x; y -= other.y; z -= other.z; return (*this); }

	Vector3 operator*(const Vector3& other) const { return Vector3(x * other.x, y * other.y, z * other.z); }
	const Vector3& operator*=(const Vector3& other) { x *= other.x; y *= other.y; z *= other.z; return (*this); }

	Vector3 operator*(float scalar) const { return Vector3(x * scalar, y * scalar, z * scalar); }
	const Vector3& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return (*this); }    

	Vector3 operator/(float scalar) const;
	const Vector3& operator/=(float scalar);

	float Length() const;
	float SquaredLength() const { return (x * x + y * y + z * z); }

	float DotProduct(const Vector3& other) const { return (x * other.x + y * other.y + z * other.z); }
	void Normalize();            

	float Unitize(float tolerance = 1e-06f );

	Vector3 CrossProduct(const Vector3& other) const { return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); }
	Vector3 UnitCrossProduct(const Vector3& other) const;

public:
	union
	{
		struct
		{
			float  x;
			float  y;
			float  z;
		};

		struct
		{
			float  comp[3];
		};
	};

	static const Vector3 ZERO;
	static const Vector3 ONE;
	static const Vector3 X_AXIS;
	static const Vector3 Y_AXIS;
	static const Vector3 Z_AXIS;
};

#endif //__VECTOR3_H__
#ifndef VECTOR3D_H
#define VECTOR3D_H

// VECTOR3D
//   3D vector.
class VECTOR3D
{
public:
	VECTOR3D()
	{
		x = y = z = 0.0f;
	}

	VECTOR3D(float x,float y,float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	VECTOR3D(const VECTOR3D &rhs)
	{
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
	}

	bool operator== (const VECTOR3D &vec) const
	{
		if((x<(vec.x-EPSILON))||(x>(vec.x+EPSILON)))
			return false;
		if((y<(vec.y-EPSILON))||(y>(vec.y+EPSILON)))
			return false;
		if((z<(vec.z-EPSILON))||(z>(vec.z+EPSILON)))
			return false;
		return true;
	}

	bool operator!= (const VECTOR3D &vec) const
	{	
		return !((*this)==vec);
	}

	VECTOR3D operator+ (const VECTOR3D &vec) const
	{
		return VECTOR3D(x+vec.x,y+vec.y,z+vec.z);
	}

	VECTOR3D operator- (const VECTOR3D &vec) const
	{
		return VECTOR3D(x-vec.x,y-vec.y,z-vec.z);
	}

	VECTOR3D operator- () const
	{
		return VECTOR3D(-x,-y,-z);
	}	
	
	VECTOR3D operator* (float scalar) const
	{
		return VECTOR3D(x*scalar,y*scalar,z*scalar);
	}

	VECTOR3D operator/ (float scalar) const
	{
		float inv = 1/scalar;
		return VECTOR3D(x*inv,y*inv,z*inv);
	}

	void operator+= (const VECTOR3D &rhs)
	{	
		x += rhs.x;	
		y += rhs.y;	
		z += rhs.z;	
	}

	void operator-= (const VECTOR3D &rhs)
	{	
		x -= rhs.x;	
		y -= rhs.y;	
		z -= rhs.z;	
	}

	void operator*= (const float rhs)
	{	
		x *= rhs;	 
		y *= rhs;
		z *= rhs;	  
	}
 
	void operator/= (const float rhs)
	{
	  float inv = 1/rhs;
		x *= inv;  
		y *= inv; 
		z *= inv;	
  }

  operator float* () const 
  {
	  return (float*) this;
  }

  operator const float* () const 
  {
	  return (const float*) this;  
  }
 
	void Set(float x,float y,float z)
	{ 
		this->x = x;
		this->y = y;
		this->z = z; 
	}

	void SetZero()
	{  
		x = y = z = 0.0f; 
	}

	void SetMin()
	{
		x = y = z = -FLT_MAX;
	}

	void SetMax()
	{
		x = y = z = FLT_MAX;
	}

	float* GetElements()
	{
		return &x;
	}

	float DotProduct(const VECTOR3D &vec) const
	{
		return ((vec.x*x)+(vec.y*y)+(vec.z*z));
	}

	VECTOR3D CrossProduct(const VECTOR3D &vec) const
	{
		return VECTOR3D((y*vec.z)-(z*vec.y),
			              (z*vec.x)-(x*vec.z),
			              (x*vec.y)-(y*vec.x));
	}

	float GetLength() const
	{
		return sqrt((x*x)+(y*y)+(z*z));
	}

	float GetSquaredLength() const
	{
		return ((x*x)+(y*y)+(z*z));
	}

	float Distance(const VECTOR3D &vec) const
	{
		return (*this-vec).GetLength();
	}

	void Normalize()
	{
		float length = sqrt((x*x)+(y*y)+(z*z));
		if(length==0.0f)
			return;
		float inv = 1/length;
		x *= inv;
		y *= inv;
		z *= inv;
	}

	VECTOR3D GetNormalized() const
	{
		VECTOR3D result(*this);
		result.Normalize();
		return result;
	}	

	void Floor()
	{
		x = floor(x);
		y = floor(y);
		z = floor(z);
	}

	VECTOR3D GetFloored()
	{
		VECTOR3D result(*this);
		result.Floor();
		return result;
	}

	float x,y,z;

};

#endif
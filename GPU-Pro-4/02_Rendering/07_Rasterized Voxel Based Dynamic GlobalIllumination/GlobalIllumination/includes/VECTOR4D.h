#ifndef VECTOR4D_H
#define VECTOR4D_H

// VECTOR4D
//   4D vector.
class VECTOR4D
{
public:
	VECTOR4D()
	{
		x = y = z = w = 0.0f;
	}

	VECTOR4D(float x,float y,float z,float w)	
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}	

	VECTOR4D(const VECTOR4D &rhs)
	{
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
		w = rhs.w;
	}

	VECTOR4D(const VECTOR3D &rhs)
	{
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
		w = 1.0f;
	}
	
	bool operator== (const VECTOR4D &vec) const
	{
		if((x<(vec.x-EPSILON))||(x>(vec.x+EPSILON)))
			return false;
		if((y<(vec.y-EPSILON))||(y>(vec.y+EPSILON)))
			return false;
		if((z<(vec.z-EPSILON))||(z>(vec.z+EPSILON)))
			return false;
		if((w<(vec.w-EPSILON))||(w>(vec.w+EPSILON)))
			return false;
		return true;
	}

	bool operator!= (const VECTOR4D &vec) const
	{	
		return !((*this)==vec);
	}

	VECTOR4D operator+ (const VECTOR4D &vec) const
	{	
		return VECTOR4D(x+vec.x,y+vec.y,z+vec.z,w+vec.w);	
	}

	VECTOR4D operator- (const VECTOR4D &vec) const
	{	
		return VECTOR4D(x-vec.x,y-vec.y,z-vec.z,w-vec.w);	
	}

	VECTOR4D operator- () const 
	{
		return VECTOR4D(-x,-y,-z,-w);
	}

	VECTOR4D operator* (const float scalar) const
	{	
		return VECTOR4D(x*scalar,y*scalar,z*scalar,w*scalar);	
	}

	VECTOR4D operator/ (const float scalar) const
	{	
		float inv = 1/scalar;
		return VECTOR4D(x*inv,y*inv,z*inv,w*inv);	
	}

	void operator+= (const VECTOR4D &rhs)
	{	
		x += rhs.x; 
		y += rhs.y; 
		z += rhs.z; 
		w += rhs.w;	
	}

	void operator-= (const VECTOR4D &rhs)
	{	
		x -= rhs.x;  
		y -= rhs.y;
		z -= rhs.z; 
		w -= rhs.w;	
	}

	void operator*= (float rhs)
	{	
		x *= rhs; 
		y *= rhs; 
		z *= rhs; 
		w *= rhs;	
	}

	void operator/= (float rhs)
	{	
    float inv = 1/rhs;
		x *= inv; 
		y *= inv; 
		z *= inv; 
		w *= inv;	
	}

	operator float* () const 
	{
		return (float*) this;
	}

	operator const float* () const 
	{
		return (const float*) this;
	}

	void Set(float x,float y,float z,float w)
	{	
		this->x = x;	
		this->y = y;	
		this->z = z; 
		this->w = w;	
	}

	void Set(const VECTOR3D &vec)
	{
    x = vec.x;
    y = vec.y;
		z = vec.z;
		w = 1.0f;
	}

	void SetZero()
	{ 
		x = y = z = w = 0.0f; 
	}

	float* GetElements()
	{
		return &x;
	}

	float x,y,z,w;

};

#endif
#ifndef VECTOR2D_H
#define VECTOR2D_H

// VECTOR2D
//   2D vector.
class VECTOR2D
{
public:
	VECTOR2D()
	{
		x = y = 0.0f;
	}

	VECTOR2D(float x,float y)
	{
		this->x = x;
		this->y = y;
	}

	VECTOR2D(const VECTOR2D &rhs)
	{
		x = rhs.x;
		y = rhs.y;
	}

	bool operator== (const VECTOR2D &vec) const
	{
		if((x<(vec.x-EPSILON))||(x>(vec.x+EPSILON)))
			return false;
		if((y<(vec.y-EPSILON))||(y>(vec.y+EPSILON)))
			return false;
		return true;
	}

	bool operator!= (const VECTOR2D &vec) const
	{	
		return !((*this)==vec);
	}

	VECTOR2D operator+ (const VECTOR2D &vec) const
	{
		return VECTOR2D(x+vec.x,y+vec.y);
	}

	VECTOR2D operator- (const VECTOR2D &vec) const
	{
		return VECTOR2D(x-vec.x,y-vec.y);
	}

	VECTOR2D operator- () const
	{
		return VECTOR2D(-x,-y);
	}

	VECTOR2D operator* (float scalar) const
	{
		return VECTOR2D(x*scalar,y*scalar);
	}

	VECTOR2D operator/ (float scalar) const
	{	
		float inv = 1/scalar;
	  return VECTOR2D(x*inv,y*inv);	
	}

	void operator+= (const VECTOR2D &rhs)
	{	
		x += rhs.x;	
		y += rhs.y;
	}

	void operator-= (const VECTOR2D &rhs)
	{	
	  x -= rhs.x;	
	  y -= rhs.y;
	}

	void operator*= (float rhs)
	{	
	  x *= rhs;	
		y *= rhs;	
	}

	void operator/= (float rhs)
	{	
	  float inv = 1/rhs;
	 	x *= inv; 
		y *= inv;	
	}

	operator float* () const 
	{
		return (float*) this;
	}

	operator const float* () const 
	{
		return (const float*) this;
	}

	void Set(float x,float y)
	{
		this->x = x;
		this->y = y; 
	}

	void SetZero()
	{  
		x = y = 0.0f; 
	}

	void SetMin()
	{
		x = y = -FLT_MAX;
	}

	void SetMax()
	{
		x = y = FLT_MAX;
	}

	float* GetElements()
	{
		return &x;
	}

	float x,y;
	
};

#endif

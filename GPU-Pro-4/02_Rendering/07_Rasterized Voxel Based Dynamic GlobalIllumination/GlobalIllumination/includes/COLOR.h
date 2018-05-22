#ifndef COLOR_H
#define COLOR_H

// COLOR
//  RGBA color.
class COLOR
{
public:
	COLOR()
	{
		r = g = b = a = 0.0f;
	}

	COLOR(float r,float g,float b,float a=1.0f)
	{
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}

	COLOR(const COLOR &rhs)
	{
		r = rhs.r;
		g = rhs.g;
		b = rhs.b;
		a = rhs.a;
	}
	
	bool operator== (const COLOR &color) const
	{
		if((r<(color.r-EPSILON))||(r>(color.r+EPSILON)))
			return false;
		if((g<(color.g-EPSILON))||(g>(color.g+EPSILON)))
			return false;
		if((b<(color.b-EPSILON))||(b>(color.b+EPSILON)))
			return false;
		if((a<(color.a-EPSILON))||(a>(color.a+EPSILON)))
			return false;
		return true;
	}

	bool operator!= (const COLOR &color) const
	{	
		return !((*this)==color);
	} 
	
	operator float* () const 
	{
		return (float*) this;
	}
	operator const float* () const 
	{
		return (const float*) this;
	}

	void Set(float r,float g,float b,float a=1.0f)
	{
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}

	float r,g,b,a;
	
};

#endif
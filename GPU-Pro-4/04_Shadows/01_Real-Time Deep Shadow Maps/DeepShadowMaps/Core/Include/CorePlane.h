#pragma once

#include "CoreVector3.h"

class CorePlane;

// Normalizes the Plane
CorePlane CorePlaneNormalize(CorePlane &pIn);

// Calcs the DotProduct of a Plane and a position-Vector
float CorePlaneDotNormal(CorePlane &pIn, CoreVector3 &vIn);

// Calcs the DotProduct of a Plane and a direction-Vector
float CorePlaneDotCoords(CorePlane &pIn, CoreVector3 &vIn);

// Transforms a Plane
CorePlane CorePlaneTransform(CorePlane &pIn, CoreMatrix4x4 &mIn);

class CorePlane
{
	// Variablen
public:
	union
	{
		struct 
		{
			CoreVector3 n;
			float d;
		};
		struct
		{
			float a,b,c;
			float d;
		};
		float arr[4];
	};

	// Constructors
	inline CorePlane()																							{};
	inline CorePlane(const float _a, const float _b, const float _c, const float _d) : a(_a), b(_b), c(_c), d(_d)	{};
	inline CorePlane(CoreVector3 _n, float _d) : n(_n), d(_d)	{};
	CorePlane(CoreVector3 &vPoint, CoreVector3 &vNormal);
	CorePlane(CoreVector3 &v1, CoreVector3 &v2, CoreVector3 &v3);


	// Operators
	inline bool operator == (const CorePlane *p2) const {if(a != p2->a) return false; if(b != p2->b) return false; if(c != p2->c) return false; return d == p2->d;}
	inline bool operator != (const CorePlane *p2) const {if(a != p2->a) return true; if(b != p2->b) return true; if(c != p2->c) return true; return d != p2->d;}

	// inline
	inline void NormalizeThis()								{ *this = CorePlaneNormalize(*this); }
	inline CorePlane Normalize(CorePlane &pOut)				{ return CorePlaneNormalize(*this); }
	inline float DotNormal(CoreVector3 &vIn)				{ return CorePlaneDotNormal(*this, vIn); }
	inline float DotCoords(CoreVector3 &vIn)				{ return CorePlaneDotCoords(*this, vIn); }
	inline void TransformThis(CoreMatrix4x4 &mIn)				{ *this = CorePlaneTransform(*this, mIn); }
	inline CorePlane Transform(CoreMatrix4x4 &mIn)				{ return CorePlaneTransform(*this, mIn); }
};
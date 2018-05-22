#include "CorePlane.h"
#include "CoreMatrix4x4.h"

// Normalizes the Plane
CorePlane CorePlaneNormalize(CorePlane &pIn)
{
	CorePlane pOut;	
	float length = pIn.n.Length();
	pOut.n = pIn.n / length;
	pOut.d = pIn.d * length;
	return pOut;
}

// Calcs the DotProduct of a Plane and a position-Vector
float CorePlaneDotNormal(CorePlane &pIn, CoreVector3 &vIn)
{
	return CoreVector3Dot(pIn.n, vIn);
}

// Calcs the DotProduct of a Plane and a direction-Vector
float CorePlaneDotCoords(CorePlane &pIn, CoreVector3 &vIn)
{
	return CoreVector3Dot(pIn.n, vIn) + pIn.d;
}

// Transforms a Plane
CorePlane CorePlaneTransform(CorePlane &pIn, CoreMatrix4x4 &mIn)
{
	float a = pIn.a * mIn._11 + pIn.b * mIn._21 + pIn.c * mIn._31;
	float b = pIn.a * mIn._12 + pIn.b * mIn._22 + pIn.c * mIn._32;
	float c = pIn.a * mIn._13 + pIn.b * mIn._23 + pIn.c * mIn._33;

	return CorePlane(a, b, c, pIn.d - (a * mIn._41 + b * mIn._42 + c * mIn._43));
}

// Constructor with Point/Normal
CorePlane::CorePlane(CoreVector3 &vPoint, CoreVector3 &vNormal)						
{
	n = vNormal.Normalize();
	d = -n.x * vPoint.x - n.y * vPoint.y - n.z * vPoint.z;
}

// Constructor with 3 Points
CorePlane::CorePlane(CoreVector3 &v1, CoreVector3 &v2, CoreVector3 &v3)	
{
	*this = CorePlane(v1, CoreVector3Cross(v3 - v2, v1 - v2));
}

#pragma once


#include "types.h"
#include <essentials/stl.h>

#include <iomanip>


inline ostream& operator << (ostream& os, const NMath::Vector2& v)
{
	os << "[ " << v.x << ", " << v.y << " ]";
	return os;
}


inline ostream& operator << (ostream& os, const NMath::Vector3& v)
{
	os << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
	return os;
}


inline ostream& operator << (ostream& os, const NMath::Matrix& m)
{
	os << fixed << setprecision(3) <<
		"[ " << m.m[0][0] << ", " << m.m[0][1] << ", " << m.m[0][2] << ", " << m.m[0][3] << " ]" << endl <<
		"[ " << m.m[1][0] << ", " << m.m[1][1] << ", " << m.m[1][2] << ", " << m.m[1][3] << " ]" << endl <<
		"[ " << m.m[2][0] << ", " << m.m[2][1] << ", " << m.m[2][2] << ", " << m.m[2][3] << " ]" << endl <<
		"[ " << m.m[3][0] << ", " << m.m[3][1] << ", " << m.m[3][2] << ", " << m.m[3][3] << " ]";
	return os;
}

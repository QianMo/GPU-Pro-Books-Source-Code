#include "DXUT.h"
#include "D3DXVectorOperators.h"

D3DXVECTOR3 operator*(const D3DXVECTOR3& a, const D3DXVECTOR3& b)
{
	return D3DXVECTOR3(a.x * b.x, a.y * b.y, a.z * b.z);
}

D3DXVECTOR3 operator/(const D3DXVECTOR3& a, const D3DXVECTOR3& b)
{
	return D3DXVECTOR3(a.x / b.x, a.y / b.y, a.z / b.z);
}

D3DXVECTOR3 reflect(const D3DXVECTOR3& d, const D3DXVECTOR3& n)
{
	D3DXVECTOR3 nodd = n * D3DXVec3Dot(&d, &n) * (-2);
	return d + nodd;
}


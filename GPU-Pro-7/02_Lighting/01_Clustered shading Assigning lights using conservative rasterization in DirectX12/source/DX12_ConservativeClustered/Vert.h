#pragma once
#include "Types.h"
#include <SimpleMath.h>

using namespace DirectX::SimpleMath;

//Temp vec3i8
struct vec4i8
{
	int8 x;
	int8 y;
	int8 z;
	int8 w;
};

struct Vert
{
	Vert(){}
	Vert(Vector3 p_Pos, Vector3 p_Normal, Vector2 p_TexC)
		: pos(p_Pos),
		normal(p_Normal),
		texc(p_TexC){}

	Vector3 pos;
	Vector3 normal;
	Vector2 texc;
};

struct VertP
{
	VertP(){}
	VertP(Vector3 p_Pos)
		: pos(p_Pos){}

	Vector3 pos;
};

struct VertP8
{
	VertP8(){}
	VertP8(vec4i8 p_Pos)
		: pos(p_Pos){}
	VertP8(Vector3 p_Pos)
	{
		pos.x = (p_Pos.x == 1.0f) ? 127 : (int)(p_Pos.x  * 128.0f);
		pos.y = (p_Pos.y == 1.0f) ? 127 : (int)(p_Pos.y  * 128.0f);
		pos.z = (p_Pos.z == 1.0f) ? 127 : (int)(p_Pos.z  * 128.0f);
		pos.w = 0;
	}

	vec4i8 pos;
};


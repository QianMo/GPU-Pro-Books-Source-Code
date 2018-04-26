/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

//***********************************************************
// this file contains the pre-implemented module operations
//***********************************************************

#ifndef MODULEOPERATIONS_FX
#define MODULEOPERATIONS_FX

#include "modules.fx"

// NOTE: the following typedef is included at modules.fx
// typedef ... PositionType;

unsigned int generationLevel;

//*********************************************************************************
// Basic methods used in rules
//	- e.g. resizing: successor's size is the parents size rescaled with a constant
//	- NOTE: module = current successor, parent = predecessor
//*********************************************************************************


//***************************************************************
// Termination : terminates the generation of the current module
//***************************************************************
#ifdef __GR_TERMINATION_DEFINED
void module_terminate( inout Module module )
{
	module.terminated = 1;
}
#endif


//****************************************************************************************************
// "Random" : creates a random value between 0 and 1
//	- NOTE: its clearly a hack, but it should work, because it wont be evaluated in continuous points
//  - use your favourite random function (e.g. Perlin noise, linear congruential functions etc.) here
//****************************************************************************************************

#define __GR_RND_FLOAT_SCALER		10000			// assumption: distance between coordinates is greater than 1/this
#define __GR_RND_MODULO				32

// helper function: returns x % __GR_RND_FLOAT_SCALER, also works properly for x < 0
int modulo( int x )
{
	const int vNeg = (abs(x+__GR_RND_FLOAT_SCALER))%__GR_RND_FLOAT_SCALER;
	const int vPos = x%__GR_RND_FLOAT_SCALER;
	return lerp( vNeg, vPos, (sign(x)+1)*0.5 );
}

float module_random( Module module )
{
	float output = 0;
#ifdef __GR_POS_DEFINED
	int u = floor( module.position.x );
	int v = floor( module.position.y );
	int w = floor( module.position.z );
	int uFrac = floor( module.position.x * __GR_RND_FLOAT_SCALER );
	int vFrac = floor( module.position.y * __GR_RND_FLOAT_SCALER );
	int wFrac = floor( module.position.z * __GR_RND_FLOAT_SCALER );
	u += uFrac;
	v += vFrac;
	w += wFrac;
	
	v = 36969*(v & 65535) + (v >> 16);
	u = 18000*(u & 65535) + (u >> 16);
	u += (v << 16); // first random number
	
	w = 36969*(w & 65535) + (w >> 16);
	v = 18000*(v & 65535) + (v >> 16);
	v += (w << 16); // second
	
	u = abs(u+v) % __GR_RND_MODULO;
	
	output = float(u) / float(__GR_RND_MODULO);
#endif
	return output;
}

float module_random_select( Module module, uint levels, uint thisLevel, float value1, float value2 )
{
	float random = module_random(module);
	if ( (1/(float)levels*(thisLevel-1) <= random) && (random < 1/(float)levels*thisLevel) )
		return value1;
	return value2;
}

//********************************************************
// Resizing : child size is a scaled value of parent size
//********************************************************

#ifdef __GR_SIZE_DEFINED
void module_resize( inout Module module, in Module parent, float size )
{
	module.size = parent.size * size;
}
#endif

#ifdef __GR_SIZE3_DEFINED
void module_resize_x( inout Module module, in Module parent, float size )
{
	module.size.x = parent.size.x * size;
}

void module_resize_y( inout Module module, in Module parent, float size )
{
	module.size.y = parent.size.y * size;
}

void module_resize_z( inout Module module, in Module parent, float size )
{
	module.size.z = parent.size.z * size;
}

void module_resize3( inout Module module, in Module parent, float size_x, float size_y, float size_z )
{
	module.size.x = parent.size.x * size_x;
	module.size.y = parent.size.y * size_y;
	module.size.z = parent.size.z * size_z;
}
#endif

#ifdef __GR_POS_DEFINED
//*****************************************************************
// Positioning : child is positioned from the parent as the origin
//*****************************************************************

void module_move( inout Module module, in Module parent, float move_x, float move_y, float move_z )
{
	float3 move = float3(move_x,move_y,move_z);
	module.position = parent.position;
	module.position.xyz = parent.position.xyz + move;
}

void module_move_x( inout Module module, in Module parent, float move )
{
	module.position.x = parent.position.x + move;
}

void module_move_y( inout Module module, in Module parent, float move )
{
	module.position.y = parent.position.y + move;
}

void module_move_z( inout Module module, in Module parent, float move )
{
	module.position.z = parent.position.z + move;
}

//********************************************************
// Rotation: child is rotated from the parent orientation
//********************************************************

#ifdef __GR_ORIENTATION_DEFINED

#define __QUAT_S(q)	q.x
#define __QUAT_D(q)	q.yzw

void multQuaternion( inout float4 q1, in float4 q2 )
{
	__QUAT_S(q1) = __QUAT_S(q1) * __QUAT_S(q2) - dot( __QUAT_D(q1), __QUAT_D(q2) );
	__QUAT_D(q1) = __QUAT_S(q1) * __QUAT_D(q2) + __QUAT_S(q2) * __QUAT_D(q1) + cross( __QUAT_D(q1), __QUAT_D(q2) );
}

#ifndef PI
#define PI 3.14159265
#endif 
#define __ANGLE_TO_RAD(angle)	((angle)* PI / 180.0 )

void multQuaternionVec( inout float4 q1, float x, float y, float z, float angle )
{
	float4 q2;
	float angleRadHalf = __ANGLE_TO_RAD(angle) / 2.0;
	__QUAT_S(q2) = cos( angleRadHalf );
	__QUAT_D(q2) = float3(x,y,z) * sin( angleRadHalf );
	
	__QUAT_S(q1) = __QUAT_S(q1)*__QUAT_S(q2) - dot(__QUAT_D(q1),__QUAT_D(q2));
	__QUAT_D(q1) = __QUAT_S(q1)*__QUAT_D(q2) + __QUAT_S(q2)*__QUAT_D(q1) + cross(__QUAT_D(q1),__QUAT_D(q2));
	
	float len = sqrt (__QUAT_S(q1)*__QUAT_S(q1) + dot(__QUAT_D(q1),__QUAT_D(q1)));
	__QUAT_S(q1) = __QUAT_S(q1) / len;
	__QUAT_D(q1) = __QUAT_D(q1) / len;
}

void quaternionRotate( inout float3 vec, in float4 q )
{
	float3 output;
	float t2 = q.x * q.y;
	float t3 = q.x * q.z;
	float t4 = q.x * q.w;
	float t5 = -q.y * q.y;
	float t6 = q.y * q.z;
	float t7 = q.y * q.w;
	float t8 = -q.z * q.z;
	float t9 = q.z * q.w;
	float t10 = -q.w * q.w;
	output.x = 2*( (t8 + t10)*vec.x + (t6 -  t4)*vec.y + (t3 + t7)*vec.z ) + vec.x;
	output.y = 2*( (t4 +  t6)*vec.x + (t5 + t10)*vec.y + (t9 - t2)*vec.z ) + vec.y;
	output.z = 2*( (t7 -  t3)*vec.x + (t2 +  t9)*vec.y + (t5 + t8)*vec.z ) + vec.z;
	vec = output;
}

void module_rotate_x( inout Module module, in Module parent, float angle )
{
	module.orientation = parent.orientation;
	multQuaternionVec( module.orientation, 1.0, 0.0, 0.0, angle );
}

void module_rotate_y( inout Module module, in Module parent, float angle )
{
	module.orientation = parent.orientation;
	multQuaternionVec( module.orientation, 0.0, 1.0, 0.0, angle );
}

void module_rotate_z( inout Module module, in Module parent, float angle )
{
	module.orientation = parent.orientation;
	multQuaternionVec( module.orientation, 0.0, 0.0, 1.0, angle );
}

void module_rotate( inout Module module, in Module parent, float angle_x, float angle_y, float angle_z )
{
	module.orientation = parent.orientation;
	module_rotate_x( module, parent, angle_x );
	module_rotate_y( module, parent, angle_y );
	module_rotate_z( module, parent, angle_z );
}

#endif

//**********************************************************************************************************************
// Scaled positioning: move vector is rescaled with the parent size (can be scaled further with an additional parameter)
//**********************************************************************************************************************

void module_scaled_move( inout Module module, in Module parent, float move_x, float move_y, float move_z, float scaler = 1.0 )
{
	float3 move = float3(move_x,move_y,move_z);
#ifdef __GR_ORIENTATION_DEFINED
	float3 move2 = move;
	quaternionRotate( move2, parent.orientation );
	module.position.xyz = parent.position.xyz + parent.size * move2 * scaler;
#else
	module.position = parent.position;
	module.position.xyz = parent.position.xyz + parent.size * move * scaler;
#endif
}

void module_scaled_move_x( inout Module module, in Module parent, float move, float scaler = 1.0 )
{
#ifdef __GR_ORIENTATION_DEFINED
	module_scaled_move( module, parent, move, 0, 0, scaler );
#else
	module.position.x = parent.position.x + parent.size * move * scaler;
#endif
}

void module_scaled_move_y( inout Module module, in Module parent, float move, float scaler = 1.0 )
{
#ifdef __GR_ORIENTATION_DEFINED
	module_scaled_move( module, parent, 0, move, 0, scaler );
#else
	module.position.y = parent.position.y + parent.size * move * scaler;
#endif
}

void module_scaled_move_z( inout Module module, in Module parent, float move, float scaler = 1.0 )
{
#ifdef __GR_ORIENTATION_DEFINED
	module_scaled_move( module, parent, 0, 0, move, scaler );
#else
	module.position.z = parent.position.z + parent.size * move * scaler;
#endif
}
#endif

//***********************************
// Code for handling time dependency
//***********************************
cbuffer TimeDependency
{
	float currentTime;
};

float dummyVariable;

//****************
// LoD management
//****************

cbuffer Camera
{
	float3 cameraPos;
	float3 unitViewVector;
	float fov;
	float aspect;
	int2 screenResolution;
}

// upper bound for the object's size in pixels
#ifdef __GR_SIZE_DEFINED
const float pixelSizeBoundScaler = 1.0f;
float module_pixelSize( in Module module )
{
	float distance = dot(module.position - cameraPos, unitViewVector);
	float denom = 1.0f/((distance - module.size)*tan(fov*0.5f));
	return pixelSizeBoundScaler * (screenResolution.x*module.size * denom)*(screenResolution.y*module.size * denom);
}
#endif

#endif
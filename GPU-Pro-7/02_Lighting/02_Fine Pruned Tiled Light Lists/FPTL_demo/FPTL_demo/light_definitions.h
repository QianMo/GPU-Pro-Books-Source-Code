#ifndef __LIGHT_DEFINITIONS_H__
#define __LIGHT_DEFINITIONS_H__


#include "shader_base.h"


#define MAX_NR_LIGHTS_PER_CAMERA		1024

#define LEFT_HAND_COORDINATES
#define VIEWPORT_SCALE_Z				1.0

unistruct cbBoundsInfo	_CB_REGSLOT(b0)
{
	Mat44 g_mProjection;
	Mat44 g_mInvProjection;
	Mat44 g_mScrProjection;
	Mat44 g_mInvScrProjection;

	unsigned int g_iNrVisibLights;
	Vec3 g_vStuff;
};


// Light types
#define MAX_TYPES					5

#define SPOT_CIRCULAR_LIGHT			0
#define WEDGE_LIGHT					1
#define SPHERE_LIGHT				2
#define CAPSULE_LIGHT				3
#define BOX_LIGHT					4


struct SFiniteLightData
{
	 // setup constant buffer
    float fAttenMapIndex;
    float fInvRange;
    float fPenumbra;
    float fInvUmbraDelta;
    
    Vec3 vLpos;
    float fLightIntensity;
    
    Vec3 vBoxAxisX;
    float fInverseLightIntensity;
    
    Vec3 vLdir;
    float fSegLength;
    
    Vec3 vBoxAxisZ;
	float	fPad0;
    
    Vec4 vBoxInnerDist;
	
    Vec4 vBoxInvRange;
    
    Vec4 vLambWeights;
    
    float	fMaxSpec;
	float	fNearRadiusOverRange_LP0;
	float	fSphRadiusSq;
	unsigned int uLightType;

	Vec3	vCol;
	float	fPad1;
};

struct SFiniteLightBound
{
	Vec3 vBoxAxisX;
	Vec3 vBoxAxisY;
	Vec3 vBoxAxisZ;
	Vec3 vCen;
	Vec2 vScaleXZ;
	float fRadius;
};


#endif
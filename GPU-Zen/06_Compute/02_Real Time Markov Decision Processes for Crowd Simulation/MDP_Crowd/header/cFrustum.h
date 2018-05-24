/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/

#pragma once
#include "cMacros.h"
#include "cProjectionManager.h"
#include <glm/glm.hpp>

using namespace glm;

//=======================================================================================

#ifndef __FRUSTUM_PLANE
#define __FRUSTUM_PLANE

class FrustumPlane
{
public:
	FrustumPlane( vec3& v1, vec3& v2, vec3& v3 )
	{
		set3Points( v1, v2, v3 );
	};
	FrustumPlane(){};
	~FrustumPlane(){};

	void set3Points( vec3& v1, vec3& v2, vec3& v3 )
	{
		vec3 A( v3.x - v2.x, v3.y - v2.y, v3.z - v2.z );
		vec3 B( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z );
		N = glm::normalize( cross( A, B ) );
		P = v2;
		D = -dot( N , P );
	}

	void setNormalAndPoint( vec3& normal, vec3& point )
	{
		N = glm::normalize( normal );
		P = point;
		D = -dot( N, P );
	}

	void setCoefficients( float a, float b, float c, float d )
	{
		N           = vec3( a, b, c );
		float len	= length( N );
		N = vec3( a / len, b / len, c / len );
		D			= d / len;
	}

	float distance( vec3 &p )
	{
		return (D + dot( N, p ));
	}

public:
	vec3	N;
	vec3	P;
	float	D;
};
#endif

//=======================================================================================

#ifndef __FRUSTUM
#define __FRUSTUM

class Frustum
{
public:
						Frustum( int type, float fov, float nearD, float farD );
						~Frustum( void );

	enum			    FRUSTUM_TYPE{ NONE, GEOMETRIC, RADAR };
	enum			    FRUSTUM_HIT{ OUTSIDE, INTERSECT, INSIDE };
	int					pointInFrustum( vec3& p );
	int					sphereInFrustum( vec3& p, float radius );
	int					boxInFrustum( vec3& box_center, vec3& box_halfdiag );
	void				setPlanes( vec3& P, vec3& L, vec3& U );
	void				setFovY( float fov );
	void				updateRatio( void );
	void				setNearD( float nearD );
	void				setFarD( float farD );
	void				drawPoints( void );
	void				drawLines( void );
	void				drawPlanes( void );
	void				drawNormals( void );
	unsigned int		getCulledCount( void );
	void				incCulledCount( void );
	void				resetCulledCount( void );
	float				getNearD( void );
	float				getFarD( void );
	float				getFovY( void );
	float				getNearW( void );
	float				getNearH( void );
	float				getFarW( void );
	float				getFarH( void );
	float				getRatio( void );
	vec3&			    getX( void );
	vec3&			    getY( void );
	vec3&			    getZ( void );
	vec3			    X;
	vec3			    Y;
	vec3			    Z;
	float				RATIO;
	float				TANG;

private:
	enum			    FRUSTUM_PLANE{ TOP, BOTTOM, LEFT, RIGHT, NEARP, FARP };
	unsigned int		culled_count;
	int					vfc_type;
	vec3			    NTL;	// NEAR_TOP_LEFT
	vec3			    NTR;	// NEAR_TOP_RIGHT
	vec3			    NBL;	// NEAR_BOTTOM_LEFT
	vec3			    NBR;	// NEAR_BOTTOM_RIGHT
	vec3			    FTL;	// FAR_TOP_LEFT
	vec3			    FTR;	// FAR_TOP_RIGHT
	vec3			    FBL;	// FAR_BOTTOM_LEFT
	vec3			    FBR;	// FAR_BOTTOM_RIGHT
	vec3			    CAM_POS;
	vec3			    near_center;
	vec3			    far_center;
	float				near_distance;
	float				far_distance;
	float				FOVY;
	float				near_width;
	float				near_height;
	float				far_width;
	float				far_height;
	float				sphereFactorX;
	float				sphereFactorY;
	float				fR;
	float				fG;
	float				fB;
	FrustumPlane		frustum_planes[6];
	ProjectionManager*	proj_man;
};

#endif

//=======================================================================================


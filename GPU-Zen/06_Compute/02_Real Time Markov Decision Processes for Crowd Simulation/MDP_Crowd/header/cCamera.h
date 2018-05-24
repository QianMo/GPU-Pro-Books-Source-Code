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
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>

#include "cMacros.h"
#include <GL/glu.h>
#include "cFrustum.h"

#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <string>
#include <math.h>

#ifndef CAMERA_ROTATE_AXIS
	#define CAMERA_ROTATE_AXIS 1
#endif

//=======================================================================================

#ifndef __CAMERA
#define __CAMERA

class Camera
{
public:
						Camera( unsigned int	id,
								int				type,
								int				frustum_type );
						~Camera( void );

	enum			    CAM_TYPE{ FIXED, FREE };
	enum			    CAM_STEREO{ NO_STEREO, ACTIVE_STEREO, ANAGLYPH_STEREO };
	void				setPosition( vec3& pos );
	void				setDirection( vec3& dir );
	void				setUpVec( vec3& up );
	void				setPivot( vec3& pivot );
	void				moveForward( float dist );
	void				moveBackward( float dist );
	void				moveRight( float dist );
	void				moveLeft( float dist );
	void				moveUp( float dist );
	void				moveDown( float dist );
	void				setView( void );
	void				move( float dir, float deg_angle );
	void				rotateAngle( float deg_angle, vec3& axis );
	void				moveAround( float deg_angle, vec3& axis );
	void				getWorldFrustumCoords( float* coords );
	void				calcUnitVectors( void );
	void				draw( void );
	void				readMatrices( void );
	unsigned int		getId( void );
	void				setInactive( void );
	int					getType( void );
	double				projectionMatrix[16];
	double				modelviewMatrix[16];
	vec3&			    getPosition( void );
	vec3&			    getDirection( void );
	vec3&			    getUp( void );
	Frustum*			getFrustum( void );
	float				updatePlanes( void );
	void				setEyeSeparation( float value );
	void				calcStereoInternals( int stereoMode );
	void				setLeftView( int stereoMode );
	void				setRightView( int stereoMode );

private:
	std::string			name;
	vec3			    X;
	vec3			    Y;
	vec3			    Z;
	vec3			    R;
	vec3			    position;
	vec3			    direction;
	vec3			    up;
	vec3				right;
	vec3			    pivot;
	vec3			    lookat;
	unsigned int		id;
	bool				hasModel;
	bool				hasAudio;
	bool				active;
	int					type;
	float				fR, fG, fB;

	float				eye_separation;
	float				wd2;
	float				ndfl;
	float				fLeft;
	float				fRight;
	float				top;
	float				bottom;
	float				focallength;
	Frustum*			frustum;
};

#endif

//=======================================================================================

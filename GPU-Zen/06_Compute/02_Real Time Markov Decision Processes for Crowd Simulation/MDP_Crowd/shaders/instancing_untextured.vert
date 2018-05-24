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
#extension GL_ARB_uniform_buffer_object: enable
#extension GL_EXT_gpu_shader4: enable
#extension GL_ARB_draw_instanced: enable

#define NUM_INSTANCES	2048

uniform mat4 ViewMat4x4;

attribute vec2 texCoord0;
attribute vec3 normal;

varying vec3 V;
varying vec3 N;

bindable uniform vec4 positions	[ NUM_INSTANCES ];
bindable uniform vec4 rotations	[ NUM_INSTANCES ];
bindable uniform vec4 scales	[ NUM_INSTANCES ];

mat4 cenital = mat4( 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 );
mat4 azimuth = mat4( 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 );

mat4 rotationToMatrix( vec2 rotation )
{
	float sinC = sin( rotation.x );
	float cosC = cos( rotation.x );
	float sinA = sin( rotation.y );
	float cosA = cos( rotation.y );	
	cenital[1][1] = cosC;
	cenital[1][2] = sinC;
	cenital[2][1] = -sinC;
	cenital[2][2] = cosC;
	azimuth[0][0] = cosA;
	azimuth[0][2] = -sinA;
	azimuth[2][0] = sinA;
	azimuth[2][2] = cosA;
	return azimuth * cenital;
}

void main( void )
{
	mat4 transMat4x4;
	vec3 position			= positions[ gl_InstanceID ].xyz;
	vec2 rotation			= rotations[ gl_InstanceID ].xy;
	mat4 scaleMat			= mat4( 1.0 );
	scaleMat[ 0 ][ 0 ]		= scales[ gl_InstanceID ].x;
	scaleMat[ 1 ][ 1 ]		= scales[ gl_InstanceID ].y;
	scaleMat[ 2 ][ 2 ]		= scales[ gl_InstanceID ].z;

	transMat4x4				= rotationToMatrix( rotation );
	transMat4x4[ 3 ][ 0 ]	= position.x;
	transMat4x4[ 3 ][ 1 ]	= position.y;
	transMat4x4[ 3 ][ 2 ]	= position.z;

	mat4 modelViewMat		= ViewMat4x4 * transMat4x4;
	vec4 P					= modelViewMat * scaleMat * gl_Vertex;
	V						= vec3( P );
	N						= normalize( gl_NormalMatrix * normal );
	gl_Position				= gl_ProjectionMatrix * P;
	
	vec3 vertexPosition		= P.xyz/P.w;
}


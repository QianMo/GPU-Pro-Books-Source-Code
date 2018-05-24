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

#define NUM_LIGHTS		1
#define DEG2RAD			0.01745329251994329576

//=======================================================================================

attribute vec2 texCoord0;
attribute vec3 normal;

varying vec3 lightVec[ NUM_LIGHTS ];
varying vec3 normalVec;

uniform sampler2D weightsTexture;								// Indicates how much to bend at joints.
uniform sampler2D zonesTexture;									// To know which bone we are processing.

uniform mat4 matPelvis;		
uniform mat4 matSpine;		
uniform mat4 matSpine1;		
uniform mat4 matNeck;		
uniform mat4 matHead;		
uniform mat4 matL_Clavicle;	
uniform mat4 matL_UpperArm;	
uniform mat4 matL_Forearm;	
uniform mat4 matL_Hand;		
uniform mat4 matL_Finger0;	
uniform mat4 matL_Finger1;	
uniform mat4 matR_Clavicle;	
uniform mat4 matR_UpperArm;	
uniform mat4 matR_Forearm;	
uniform mat4 matR_Hand;		
uniform mat4 matR_Finger0;	
uniform mat4 matR_Finger1;	
uniform mat4 matL_Thigh;	
uniform mat4 matL_Calf;		
uniform mat4 matL_Foot;		
uniform mat4 matL_Toe0;		
uniform mat4 matR_Thigh;	
uniform mat4 matR_Calf;		
uniform mat4 matR_Foot;		
uniform mat4 matR_Toe0;		

uniform samplerBuffer posTextureBuffer;							// Each instance's position.
uniform mat4 ViewMat4x4;

//=======================================================================================

mat4 rotationToMatrix( vec2 rotation )							// Classic 3D Rotation Matrix.
{
	mat4 cenital	= mat4( 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 );
	mat4 azimuth	= mat4( 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 );
	float sinC		= sin( rotation.x );
	float cosC		= cos( rotation.x );
	float sinA		= sin( rotation.y );
	float cosA		= cos( rotation.y );
	cenital[1][1]	=  cosC;
	cenital[1][2]	=  sinC;
	cenital[2][1]	= -sinC;
	cenital[2][2]	=  cosC;
	azimuth[0][0]	=  cosA;
	azimuth[0][2]	= -sinA;
	azimuth[2][0]	=  sinA;
	azimuth[2][2]	=  cosA;
	return cenital * azimuth;
}

//=======================================================================================

void main( void )
{
	gl_TexCoord[ 0 ].st		= texCoord0.xy;
	vec4 inputv				= texelFetchBuffer( posTextureBuffer, gl_InstanceID );	// Get current instance 'global' position.
	vec4 weightVal			= texture2DLod( weightsTexture, gl_TexCoord[0].st, 0.0 );		// Weight in [0,1].
	vec3 zoneVal			= texture2DLod( zonesTexture,   gl_TexCoord[0].st, 0.0 ).rgb;	// Current zone.
	vec4 position			= vec4( inputv.x, -30.0, inputv.y, 1.0 );				// Y is fixed to -30.0 for now...
	vec2 rotation			= vec2( 0.0, inputv.z );								// Model orientation.

	float arg				= (zoneVal.r+zoneVal.g+zoneVal.b) * 31.875; // 31.875 = 255.0/8.0
	int zone_index			= trunc( arg );

	mat4 transMat4x4		= rotationToMatrix( rotation );							// Orient whole model's instance.
	transMat4x4[ 3 ][ 0 ]	= position.x;											// Locate whole model's instance.
	transMat4x4[ 3 ][ 1 ]	= position.y;											// Locate whole model's instance.
	transMat4x4[ 3 ][ 2 ]	= position.z;											// Locate whole model's instance.
	mat4 modelViewMat		= ViewMat4x4 * transMat4x4;								// Create model's View Matrix.		

	vec4 tempva				= vec4( 0.0, 0.0, 0.0, 0.0 );

	float rem_rgb			= 1.0 - (weightVal.r + weightVal.g + weightVal.b);
	float rem_rg			= 1.0 - (weightVal.r + weightVal.g);
	float rem_gb			= 1.0 - (weightVal.g + weightVal.b);
	
	rem_rgb					= max( 0.0, rem_rgb / 3.0 );
	rem_rg					= max( 0.0, rem_rg  / 2.0 );
	rem_gb					= max( 0.0, rem_gb  / 2.0 );

	if( zone_index == 0 )
	{
		tempva			    =	(weightVal.r + rem_rgb) * matPelvis * gl_Vertex;
		tempva			   +=	(weightVal.g + rem_rgb) * matSpine  * gl_Vertex;
		tempva			   +=	(weightVal.b + rem_rgb) * matSpine1 * gl_Vertex;
	}
	else if( zone_index == 1 )
	{
		tempva			    =	(weightVal.r + rem_rg) * matNeck   * gl_Vertex;
		tempva			   +=	(weightVal.g + rem_rg) * matHead   * gl_Vertex;
	}
	else if( zone_index == 2 )
	{
		tempva			    =	(weightVal.g + rem_rgb) * matL_Clavicle * gl_Vertex;
		tempva			   +=	(weightVal.b + rem_rgb) * matL_UpperArm * gl_Vertex;
		tempva			   +=	(weightVal.r + rem_rgb) * matL_Forearm  * gl_Vertex;
	}
	else if( zone_index == 3 )
	{
		tempva			    =	matL_Hand * gl_Vertex;
	}
	else if( zone_index == 4 )
	{
		tempva			    =	matL_Finger0 * gl_Vertex;
	}
	else if( zone_index == 5 )
	{
		tempva			    =	matL_Finger1 * gl_Vertex;
	}
	else if( zone_index == 6 )
	{
		tempva			    =	(weightVal.g + rem_rgb) * matR_Clavicle * gl_Vertex;
		tempva			   +=	(weightVal.b + rem_rgb) * matR_UpperArm * gl_Vertex;
		tempva			   +=	(weightVal.r + rem_rgb) * matR_Forearm  * gl_Vertex;
	}
	else if( zone_index == 7 )
	{
		tempva			    =	matR_Hand * gl_Vertex;
	}
	else if( zone_index == 8 )
	{
		tempva			    =	matR_Finger0 * gl_Vertex;
	}
	else if( zone_index == 9 )
	{
		tempva			    =	matR_Finger1 * gl_Vertex;
	}
	else if( zone_index == 10 )
	{
		tempva			    =	(weightVal.g + rem_gb) * matL_Thigh * gl_Vertex;
		tempva			   +=	(weightVal.b + rem_gb) * matL_Calf  * gl_Vertex;
	}
	else if( zone_index == 11 )
	{
		tempva			    =	(weightVal.r + rem_rg) * matL_Foot * gl_Vertex;
		tempva			   +=	(weightVal.g + rem_rg) * matL_Toe0 * gl_Vertex;
	}
	else if( zone_index == 12 )
	{
		tempva			    =	(weightVal.g + rem_gb) * matR_Thigh * gl_Vertex;
		tempva			   +=	(weightVal.b + rem_gb) * matR_Calf  * gl_Vertex;
	}
	else if( zone_index == 13 )
	{
		tempva			    =	(weightVal.r + rem_rg) * matR_Foot * gl_Vertex;
		tempva			   +=	(weightVal.g + rem_rg) * matR_Toe0 * gl_Vertex;
	}
	else
	{
		tempva = gl_Vertex;
	}

	vec4 P					= (modelViewMat * tempva);

	gl_Position				= gl_ProjectionMatrix * P;
	normalVec				= normal;

	for( int l = 0; l < NUM_LIGHTS; l++ )
	{
		lightVec[l]			= normalize( gl_LightSource[l].position.xyz - P.xyz );
	}
}

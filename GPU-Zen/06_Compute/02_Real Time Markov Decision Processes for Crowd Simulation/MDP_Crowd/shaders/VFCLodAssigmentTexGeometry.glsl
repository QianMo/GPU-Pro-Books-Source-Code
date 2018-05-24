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

#version 120
#extension GL_ARB_texture_rectangle : require
#extension GL_EXT_geometry_shader4 : enable

#define		USING_IDS  
#define		VFC_TOLERANCE		32.0
#define		VFC_TOLERANCE_Z		32.0

#ifdef USING_IDS
	uniform sampler2DRect idTex; // this texture stores id info, not positions
#endif 

uniform sampler2DRect positions;
uniform float nearPlane;
uniform float farPlane;
uniform float tang;
uniform float ratio;
uniform float modelScale;
uniform vec3 camPos;
uniform vec3 X;
uniform vec3 Y;
uniform vec3 Z;
float lod;

bool pointInFrustum( vec3 point, out float lod )
{
	float pcz, pcx, pcy, aux;
	// Compute vector from camera position to p.
	vec3 v = point - camPos;
	// Compute and test the Z coordinate.
	pcz = dot( v, -Z );

	if( pcz > farPlane || pcz < (nearPlane - VFC_TOLERANCE) )
	{
		return false;
	}
	// Compute and test the Y coordinate.
	pcy = dot( v, Y );
	aux = pcz * tang;
	if( pcy > (aux + VFC_TOLERANCE) || pcy < -(aux + VFC_TOLERANCE) )
	{
		return false;
	}
	// Compute and test the X coordinate.
	pcx = dot( v, X );
	aux = aux * ratio;
	if( pcx > (aux + VFC_TOLERANCE_Z) || pcx < -(aux + VFC_TOLERANCE_Z) )
	{
		return false;
	}
	//Original:
	//lod = step(nearPlane, pcz) + step(farPlane/10.0, pcz) + step(farPlane/6.0, pcz);
	//Woman:
	//lod = step(nearPlane, pcz) + step(farPlane/6.0, pcz) + step(farPlane/3.0, pcz);
	//Lemmings:
	//lod = step(nearPlane, pcz) + step(farPlane/30.0, pcz) + step(farPlane/18.0, pcz);
	//lod = step(nearPlane, pcz) + step(farPlane/50.0, pcz) + step(farPlane/30.0, pcz);
	
	//lod = 3.0;
	//if( pcz < (nearPlane + 500.0) ) lod = 1.0;
	//else if( pcz < (nearPlane + 2000.0) ) lod = 2.0;
	//lod = 1.0;
	
	// 4 LODs 
	//lod = step(0.001, pcz) + step(farPlane/40.0, pcz)  + step(farPlane/12.0, pcz) + step(farPlane/4.0, pcz); 
	
	// For 3 LODs:
	// For TEASER:
	lod = step(0.001, pcz) + step(farPlane*modelScale/40.0, pcz)  + step(farPlane*modelScale/10.0, pcz); 
	// For PERFORMANCE:
	//lod = step(0.001, pcz) + step(farPlane/50.0, pcz)  + step(farPlane/30.0, pcz); 
	
	return true;
}

void main( void )
{
	lod = 0.0;
	vec2 texcoord = gl_PositionIn[0].xy;
	vec4 pos = texture2DRect( positions, texcoord );

#ifdef USING_IDS
	vec4 ids = texture2DRect(idTex, texcoord);
#endif

	if (pointInFrustum( pos.xyz, lod))
	{
		
		
#ifdef USING_IDS
		pos.x = gl_TexCoordIn[0][0].x; //glVertexId
		pos.y = ids.x; // group Id
#else
		pos.y = pos.z;
#endif
		pos.z = pos.w;
		pos.w = lod;
		gl_Position = pos;
		EmitVertex();
		EndPrimitive();
	}
}

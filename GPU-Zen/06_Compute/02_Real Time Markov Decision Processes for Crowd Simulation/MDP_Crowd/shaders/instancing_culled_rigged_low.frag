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

#define NUM_LIGHTS		1

uniform sampler2D diffuseTexture;
uniform sampler2D zonesTexture;
uniform sampler2D weightsTexture;

varying vec3 lightVec[NUM_LIGHTS];
varying vec3 normalVec;

void main( void )
{
	vec4 diffuseMaterial = texture2D( diffuseTexture, gl_TexCoord[0].st );
	//vec4 diffuseMaterial = texture2DLod( zonesTexture, gl_TexCoord[0].st, 1.0 );
	//vec4 diffuseMaterial = texture2DLod( weightsTexture, gl_TexCoord[0].st, 1.0 );
	float ambient = 0.0;
	for( int l = 0; l < NUM_LIGHTS; l++ )
	{
		ambient = max( gl_LightSource[l].ambient.r, ambient );
		float lambertFactor = max( dot( lightVec[l], normalVec ), ambient );
		lightVec[l] *= vec3( -1 );
		float lambertFactorNeg = max( dot( lightVec[l], normalVec ), ambient );
		vec4 diffuseLight = gl_LightSource[l].diffuse;
		gl_FragColor += diffuseMaterial * diffuseLight * ( lambertFactor + lambertFactorNeg );
	}
	gl_FragColor /= NUM_LIGHTS;
	gl_FragColor = diffuseMaterial;
}

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

#define NUM_LIGHTS 2

uniform sampler2D diffuseTexture;
uniform int textured;
uniform float opacity;
uniform vec3 tint;
uniform vec3 amb;

varying vec3 N;
varying vec3 V;

/*
void main( void )
{
	vec4 diffuseMaterial = vec4( tint, 1 );
	float tex_alpha = 1.0;
	float intensity = 0.0;
	vec4 final = vec4( 0.0 );

	if( textured > 0 )
	{
		vec4 tex_comp = texture2D( diffuseTexture, gl_TexCoord[0].st );
		diffuseMaterial	= tex_comp;
		tex_alpha = tex_comp.a;
	}
	vec4 ambient = diffuseMaterial * vec4( amb, 1.0 );

	for( int i = 0; i < NUM_LIGHTS; i++ )
	{
		vec3 L = normalize( vec3( gl_LightSource[i].position ) );
		intensity = max(dot(L,N), 0.0);
		final += diffuseMaterial * intensity;
	}
	final.a = opacity * tex_alpha;
 
	// Set fragment color from texture   
	gl_FragColor = ambient + final;
}
*/


void main( void )
{
	vec4 diffuseMaterial = vec4( tint, 1 );
	float tex_alpha = 1.0;

	if( textured > 0 )
	{
		vec4 tex_comp = texture2D( diffuseTexture, gl_TexCoord[0].st );
		diffuseMaterial	= tex_comp;
		tex_alpha = tex_comp.a;
	}
	vec4 ambient = diffuseMaterial * vec4( amb, 1.0 );

	vec4 diff = vec4( 0.0 );
	for( int i = 0; i < NUM_LIGHTS; i++ )
	{
		//vec3 L = normalize( gl_LightSource[i].position.xyz - V );
		vec3 L = normalize( vec3( gl_LightSource[i].position ) );
		//vec4 Idiff = diffuseMaterial * gl_FrontLightProduct[i].diffuse * max(dot(N,L), 0.0);
		//Idiff = clamp(Idiff, 0.0, 1.0);
		//vec4 Idiff = diffuseMaterial * gl_LightSource[i].diffuse * max(dot(N,L), 0.0);
		vec4 Idiff = gl_LightSource[i].diffuse * max(dot(N,L), 0.0);
		diff += Idiff;
	}
	diff *= diffuseMaterial;
	
	vec4 color		= clamp( ambient + diff, 0.0, 1.0 );
	gl_FragColor	= vec4( color.rgb, opacity * tex_alpha );
}

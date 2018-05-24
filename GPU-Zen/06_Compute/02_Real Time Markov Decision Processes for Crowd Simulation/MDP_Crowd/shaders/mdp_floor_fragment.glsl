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

#version 130
#extension GL_EXT_texture_array : enable

varying float level;
varying float intensity;
varying vec2 policoord;

uniform sampler2D layer0;
uniform sampler2D layer1;
uniform sampler2DArray	arrowsTA;

uniform float tile_width;
uniform float tile_height;

void main()
{
	float poli_col = trunc( policoord.x );
	float poli_row = trunc( policoord.y );
	vec2 texUV;
	texUV.x = (gl_TexCoord[0].s / tile_width)  - poli_col;
	texUV.y = (gl_TexCoord[0].t / tile_height) - poli_row;

	vec4 arrow = texture2DArray( arrowsTA, vec3( texUV, level ) );

	vec4 l0color = texture2D( layer0, texUV );
	vec4 l1color = texture2D( layer1, texUV );

	float factor = 1.0 - intensity / 2.0;
	if( intensity > 0.1 && intensity < 0.33 )
	{
		l1color.rb *= factor;	
	}
	else if ( intensity < 0.66 )
	{
		l1color.b *= factor;
	}
	else
	{
		l1color.gb *= factor;
	}

	vec4 color = (1.0-arrow.a)*l0color + arrow.a*l1color;
	/*
	vec4 color = vec4( 0.5 );
	if( level >=0.0 && level < 1.0 )
	{
		color = vec4( 0.0, 0.0, 0.0, 0.0 );
	}
	else if( level >= 1.0 && level < 2.0 )
	{
		color = vec4( 0.0, 0.0, 1.0, 0.0 );
	}
	else if( level >= 2.0 && level < 3.0 )
	{
		color = vec4( 0.0, 1.0, 0.0, 0.0 );
	}
	else if( level >= 3.0 && level < 4.0 )
	{
		color = vec4( 0.0, 1.0, 1.0, 0.0 );
	}
	else if( level >= 4.0 && level < 5.0 )
	{
		color = vec4( 1.0, 0.0, 0.0, 0.0 );
	}
	else if( level >= 5.0 && level < 6.0 )
	{
		color = vec4( 1.0, 0.0, 1.0, 0.0 );
	}
	else if( level >= 6.0 && level < 7.0 )
	{
		color = vec4( 1.0, 1.0, 0.0, 0.0 );
	}
	else if( level >= 7.0 && level < 8.0 )
	{
		color = vec4( 1.0, 1.0, 1.0, 0.0 );
	}
	*/

	//color.a = 1.0;
	gl_FragColor = color;
}

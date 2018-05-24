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

#extension GL_ARB_texture_rectangle : enable

#define BlendOverlayf(base, blend) 		(base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend)))
#define Blend(base, blend, funcf) 		vec3(funcf(base.r, blend.r), funcf(base.g, blend.g), funcf(base.b, blend.b))
#define BlendOverlay(base, blend) 		Blend(base, blend, BlendOverlayf)

uniform sampler2D pattern_tex;		// 0
uniform samplerRect coordinate_tex; // 1
uniform sampler2D wrinkle_tex;		// 2

void main ()
{
	
	float patternColorId = texture2D (pattern_tex, gl_TexCoord[0].st).r;

	// para female_coordinate_tex:
	// s = 0 es el renglon de la tabla de colores
	// s = 1 es el renglon de combinaciones 
	// t = 0 es la columna que da el color de la tabla de colores
	// t = 1 es la columna que da el color de las combinaciones
	// el segundo 1 es el id de la combinacion
	vec4 outfitCoordinate = texture2DRect (coordinate_tex, vec2(1.0, 3.0)) * 255.0; // expanding the range from [0,1] -> [0, 255]

	vec4 colorCoordinate1 = texture2DRect (coordinate_tex, vec2(0.0, outfitCoordinate.r));
	vec4 colorCoordinate2 = texture2DRect (coordinate_tex, vec2(0.0, outfitCoordinate.g));
	vec4 colorCoordinate3 = texture2DRect (coordinate_tex, vec2(0.0, outfitCoordinate.b));

	//vec4 finalColor = texture2DRect (coordinate_tex, vec2(0.0, patternColorId));

	// arrugas de la ropa
	vec4 foldingColor = texture2D (wrinkle_tex, gl_TexCoord[0].st);
	//foldingColor.a = 0.0;
	
	gl_FragColor = vec4(1.0,0.0,1.0,1.0);
/*
	if (patternColorId == 1.0)
	{
		gl_FragColor = colorCoordinate1 ;
		gl_FragColor.a = 1.0;
	}
	if (patternColorId == 2.0)
	{
		gl_FragColor = colorCoordinate2 ;
		gl_FragColor.a = 1.0;
	}
	if (patternColorId == 3.0)
	{
		gl_FragColor = colorCoordinate3 ;
		gl_FragColor.a = 1.0;
	}
*/

	gl_FragColor = vec4(BlendOverlay(gl_FragColor, foldingColor),gl_FragColor.a);
	
	gl_FragColor = vec4(foldingColor.a);
	//gl_FragColor = vec4 (1,0,0,1);
	//gl_FragColor = foldingColor;
}


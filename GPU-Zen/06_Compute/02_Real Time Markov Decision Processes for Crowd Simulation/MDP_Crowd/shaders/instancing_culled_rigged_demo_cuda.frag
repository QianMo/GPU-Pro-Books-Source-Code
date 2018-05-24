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

#define BlendOverlayf(base, blend) 		(base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend)))
#define Blend(base, blend, funcf) 		vec3(funcf(base.r, blend.r), funcf(base.g, blend.g), funcf(base.b, blend.b))
#define BlendOverlay(base, blend) 		Blend(base, blend, BlendOverlayf)

varying vec3 lightVec;
varying vec3 N;
varying vec3 ti;
varying vec4 normalColor;
varying float agent_id;
varying float agent_id2;
varying float agent_id4;

uniform sampler2DArray globalMT;	//SKIN, HAIR, CAP
uniform sampler2DArray torsoMT;		//TORSO, WRINKLES, PATTERN_0...PATTERN_N
uniform sampler2DArray legsMT;		//LEGS, WRINKLES, PATTERN
//uniform sampler2DArray riggingMT;
uniform float lod;
uniform float gender;
uniform float doColor;
uniform float doPatterns;
uniform float doFacial;
uniform sampler2DArray facialMT;

uniform sampler2DRect pattern_table;	//LINE, NUM_COLORS, PATTERN, SEASON.
uniform sampler2DRect color_table;		//RGBA0...RGBA10.

vec3 RGBToHSL(vec3 color);
vec3 HSLToRGB(vec3 hsl);

vec4 torso_coordinate ( sampler2DArray torso, sampler2DRect color_table, vec2 st, vec4 pData )
{
	vec4 color = vec4 ( 0.0, 0.0, 0.0, 1.0 );
	vec4 foldings = texture2DArray( torso, vec3( st, 0.0 ) );
	foldings.rgb *= 0.7;				// brightness adjustment
	
	vec4 pattern = texture2DArray( torso, vec3( st, 1.0 ) );
	float pattern_id = 0.0;
	float temp = agent_id2 * 512.0;
	if( temp < 256.0 )
	{
		pattern_id = pattern.r;
	}
	else
	{
		pattern_id = pattern.g;
	}
	color.a = 1.0 - pattern_id;			// setting alpha value to final composite
	//NO_PATTERNS:
	if( doPatterns == 0.0 )
	{
		pattern_id = 0.0;
	}
	pattern_id *= 255.0;				// expanding the range from [0,1] -> [0, 255]
	pattern_id = fmod( pattern_id, 11.0 );
	float num_patterns = 98.0;
	
	//ORIGINAL:
	float row = fmod( agent_id, num_patterns );
	//SUMMER_COLORS:
	//float row = fmod( agent_id, num_patterns/2.0 );
	//WINTER_COLORS:
	//float row = num_patterns/2.0 + fmod( agent_id, num_patterns/2.0 );

	//float row = agent_id - 50.0 * floor(agent_id/50.0);
	//color.rgb = texture2DRect( color_table, vec2( pattern_id+0.01, pData.r ) ).rgb;
	color.rgb = texture2DRect( color_table, vec2( pattern_id+0.5/11.0, row+0.5/num_patterns ) ).rgb;
	
	//SUMMER_COLORS (BRIGHTEN):
	//color.rgb += vec3( 0.05 );
	//WINTER_COLORS (DARKEN):
	//color.rgb -= vec3( 0.2 );
	
	color.rgb = BlendOverlay( foldings, color );
	
	return color;
}

vec4 legs_coordinate ( sampler2DArray legs, sampler2DRect color_table, vec2 st, vec4 pData )
{
	vec4 color = vec4 ( 0.0, 0.0, 0.0, 1.0 );
	vec4 foldings = texture2DArray( legs, vec3( st, 0.0 ) );
	foldings.rgb *= 0.7;				// brightness adjustment
	
	vec4 pattern = texture2DArray( legs, vec3( st, 1.0 ) );
	float pattern_id = pattern.r;
	color.a = 1.0 - pattern_id;			// setting alpha value to final composite
	pattern_id *= 255.0;				// expanding the range from [0,1] -> [0, 255]	
	pattern_id = fmod( pattern_id, 11.0 );
	float num_patterns = 98.0;
	float row = fmod( agent_id, num_patterns );
	//float col = fmod( agent_id+pattern_id, 11.0 );
	//color.rgb = texture2DRect( color_table, vec2( col+0.5/11.0, row+0.5/99.0 ) ).rgb;
	float col1 = fmod( agent_id+pattern_id,   11.0 );
	float col2 = fmod( agent_id+pattern_id+2, 11.0 );
	vec3 color1 = texture2DRect( color_table, vec2( col1+0.5/11.0, row+0.5/num_patterns ) ).rgb;
	vec3 color2 = texture2DRect( color_table, vec2( col2+0.5/11.0, row+0.5/num_patterns ) ).rgb;
	color.rgb = mix( color1.rgb, color2.rgb, 0.5 );
	color.rgb -= (color.rgb*0.2);
	color.rgb = BlendOverlay( foldings, color );
	return color;
}

vec3 modulation( vec3 color, vec3 hsl )
{
	color = RGBToHSL( color );
	color += hsl/6.0;
	return HSLToRGB ( color );
}

void main( void )
{
	//vec4 pattern_data		= texture2DRect( pattern_table, vec2( 0.0, 0.0 ) ); //LINE, NUM_COLORS, PATTERN, SEASON.
	vec4 pattern_data		= vec4( 0.0 );

	//vec2 uv					= gl_TexCoord[0].st * 0.2 + 0.2 * ti.st;
	vec2 uv					= gl_TexCoord[0].st * 0.2 + 0.2 * ti.st;
	vec4 diffuseMaterial	= texture2DArray( globalMT,	vec3(uv,0) );
	vec4 hairMaterial		= texture2DArray( globalMT,	vec3(uv,1) );
	vec4 capMaterial		= texture2DArray( globalMT,	vec3(uv,2) );
	//[R]SPOTS [G]EYESPOTS [B]BEARD [A]MOUSTACHE	
	vec4 faceMaterial		= texture2DArray( facialMT,	vec3(gl_TexCoord[0].st,1) );
	vec3 spots				= faceMaterial.rrr;
	vec3 eyespots			= faceMaterial.ggg;
	vec3 beard				= faceMaterial.bbb;
	vec3 moustache			= faceMaterial.aaa;
	vec4 torsoMaterial		= torso_coordinate( torsoMT,
												color_table,
												gl_TexCoord[0].st,
												pattern_data	);
	
	vec4 legsMaterial		= legs_coordinate( legsMT,	
											   color_table,
											   gl_TexCoord[0].st,
											   pattern_data		);

	diffuseMaterial.rgb	= modulation ( diffuseMaterial.rgb, vec3 ( -(ti.s * 3.0) / 180.0 * 0.5 , (ti.s * 3.0) / 180.0 * 0.5, 0.0 ) ); 

//->FACIAL_FEATURES
	if( doFacial == 1.0 )
	{
		float id_reference = agent_id4*2.0;
		if( id_reference < 1.9 )
		{
			diffuseMaterial.rgb = BlendOverlay( spots, diffuseMaterial.rgb );
		}
		else if( id_reference < 3.9 )
		{
			diffuseMaterial.rgb = BlendOverlay( eyespots, diffuseMaterial.rgb );
		}
		else if( id_reference < 5.9 )
		{
			diffuseMaterial.rgb = BlendOverlay( spots, diffuseMaterial.rgb );
			diffuseMaterial.rgb = BlendOverlay( eyespots, diffuseMaterial.rgb );
		}
		if( gender == 0.0f )
		{
			if( id_reference < 1.9 )
			{
				diffuseMaterial.rgb = BlendOverlay ( beard, diffuseMaterial.rgb );
			}
			else if( id_reference < 3.9 )
			{
				diffuseMaterial.rgb = BlendOverlay ( moustache, diffuseMaterial.rgb );
			}
		}
		// BEARDED WOMAN!!!
		else if( fmod( agent_id, 100.0 ) == 0.0 )
		{
			diffuseMaterial.rgb = BlendOverlay ( beard, diffuseMaterial.rgb );
		}
	}
//<-FACIAL_FEATURES

	diffuseMaterial.rgb = mix(diffuseMaterial.rgb, hairMaterial.rgb,	hairMaterial.a);
	diffuseMaterial.rgb = mix(diffuseMaterial.rgb, capMaterial.rgb,		capMaterial.a);
	diffuseMaterial.rgb = mix(diffuseMaterial.rgb, torsoMaterial.rgb,	torsoMaterial.a);	
	diffuseMaterial.rgb = mix(diffuseMaterial.rgb, legsMaterial.rgb,	legsMaterial.a);

	gl_FragColor = diffuseMaterial * 0.12;
	float NdotL; //cosin normal lightDir
	vec4 diffuseC;
	NdotL = max( dot( lightVec, N ), 0.12 );
	lightVec *= vec3( -1 );
	float MNdotL = max( dot( lightVec, N ), 0.12 );
	diffuseC = diffuseMaterial * (NdotL + MNdotL);
	gl_FragColor += diffuseC;
	// GRAY_COLOR:
	if( doColor == 0.0 )
	{
		gl_FragColor.rgb = vec3( NdotL + MNdotL );
	}
	//gl_FragColor += normalColor;
	gl_FragColor.a = 1.0;
	//gl_FragColor = texture2DArray( riggingMT,	vec3(gl_TexCoord[0].st, 0) );
}

//Hue, saturation, luminance
vec3 RGBToHSL(vec3 color)
{
	vec3 hsl; // init to 0 to avoid warnings ? (and reverse if + remove first part)
	
	float fmin = min(min(color.r, color.g), color.b);    //Min. value of RGB
	float fmax = max(max(color.r, color.g), color.b);    //Max. value of RGB
	float delta = fmax - fmin;             //Delta RGB value

	hsl.z = (fmax + fmin) * 0.5; // Luminance

	if (delta == 0.0)		//This is a gray, no chroma...
	{
		hsl.x = 0.0;	// Hue
		hsl.y = 0.0;	// Saturation
	}
	else                                    //Chromatic data...
	{
		if (hsl.z < 0.5)
			hsl.y = delta / (fmax + fmin); // Saturation
		else
			hsl.y = delta / (2.0 - fmax - fmin); // Saturation
		
		float halfDelta = delta / 2.0;
		float deltaR = (((fmax - color.r) / 6.0) + halfDelta) / delta;
		float deltaG = (((fmax - color.g) / 6.0) + halfDelta) / delta;
		float deltaB = (((fmax - color.b) / 6.0) + halfDelta) / delta;

		if (color.r == fmax )
			hsl.x = deltaB - deltaG; // Hue
		else if (color.g == fmax)
			hsl.x = (1.0 / 3.0) + deltaR - deltaB; // Hue
		else if (color.b == fmax)
			hsl.x = (2.0 / 3.0) + deltaG - deltaR; // Hue

		if (hsl.x < 0.0)
			hsl.x += 1.0; // Hue
		else if (hsl.x > 1.0)
			hsl.x -= 1.0; // Hue
	}
	return hsl;
}

float HueToRGB(float f1, float f2, float hue)
{
	if (hue < 0.0)
		hue += 1.0;
	else if (hue > 1.0)
		hue -= 1.0;
	float res;
	if ((6.0 * hue) < 1.0)
		res = f1 + (f2 - f1) * 6.0 * hue;
	else if ((2.0 * hue) < 1.0)
		res = f2;
	else if ((3.0 * hue) < 2.0)
		res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
	else
		res = f1;
	return res;
}

vec3 HSLToRGB(vec3 hsl)
{
	vec3 rgb;
	
	if (hsl.y == 0.0)
		rgb = vec3(hsl.z); // Luminance
	else
	{
		float f2;
		
		if (hsl.z < 0.5)
			f2 = hsl.z * (1.0 + hsl.y);
		else
			f2 = (hsl.z + hsl.y) - (hsl.y * hsl.z);
			
		float f1 = 2.0 * hsl.z - f2;
		
		rgb.r = HueToRGB(f1, f2, hsl.x + (1.0/3.0));
		rgb.g = HueToRGB(f1, f2, hsl.x);
		rgb.b= HueToRGB(f1, f2, hsl.x - (1.0/3.0));
	}
	return rgb;
}

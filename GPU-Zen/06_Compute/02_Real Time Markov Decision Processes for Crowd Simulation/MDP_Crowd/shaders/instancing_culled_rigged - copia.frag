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

#define NUM_LIGHTS		2
#define USING_IDS

vec3 RGBToHSL(vec3 color);
vec3 HSLToRGB(vec3 hsl);

vec3 modulation (vec3 color, vec3 hsl)
{
	color = RGBToHSL(color);
	color += hsl; 
	return HSLToRGB (color);
}

varying vec3 lightVec[NUM_LIGHTS];
varying vec3 N;
varying vec3 E;
varying vec3 ti;
varying vec4 normalColor;

uniform sampler2D diffuseTexture1;		//	skin
uniform sampler2D hair;		
uniform sampler2D _cap_;		
uniform sampler2D torso;		
uniform sampler2D legs;

void main( void )
{
	
#ifdef USING_IDS
	vec2 uv					= gl_TexCoord[0].st*0.2 + 0.2*ti.st;
	vec4 diffuseMaterial	= texture2D( diffuseTexture1,	uv );
	vec4 hairMaterial		= texture2D( hair,				uv );
	vec4 capMaterial		= texture2D( _cap_,				uv );
	vec4 torsoMaterial		= texture2D( torso,				uv );
	vec4 legsMaterial		= texture2D( legs,				uv );

	diffuseMaterial.rgb	= modulation ( diffuseMaterial.rgb, vec3 ( -(ti.s * 3.0) / 180.0 * 0.5 , (ti.s * 3.0) / 180.0 * 0.5, 0.0 ) ); 
	diffuseMaterial.rgb = mix(diffuseMaterial.rgb, hairMaterial.rgb,	hairMaterial.a);
	diffuseMaterial.rgb = mix(diffuseMaterial, capMaterial.rgb,		capMaterial.a);
	diffuseMaterial.rgb = mix(diffuseMaterial.rgb, torsoMaterial.rgb,	torsoMaterial.a);	
	diffuseMaterial.rgb = mix(diffuseMaterial.rgb, legsMaterial.rgb,	legsMaterial.a);
#else
	vec4 diffuseMaterial	= 1.0-texture2D( diffuseTexture1,	gl_TexCoord[0].st );
#endif

	gl_FragColor = diffuseMaterial*0.2;
	float NdotL; //cosin normal lightDir
	float HdotN; //cosin half way vector normal
	vec3 lightDir;
	vec3 halfVector;
	vec4 diffuseC;
	vec4 specularC;
	for( int l = 0; l < NUM_LIGHTS; l++ )
	{
		lightDir = normalize(vec3(gl_LightSource[l].position));
		NdotL = max(dot(N, lightDir), 0.0);
		diffuseC = diffuseMaterial * gl_LightSource[l].diffuse * NdotL;
		gl_FragColor += diffuseC;
		//gl_FragColor += normalColor;
		/*
		if (NdotL > 0.0) {
			halfVector = normalize(lightDir - normalize(E));
			HdotN = max(0.0, dot(halfVector,  N));
			specularC = gl_FrontMaterial.specular * gl_LightSource[l].specular * pow (HdotN, gl_FrontMaterial.shininess);
			gl_FragColor += specularC;
		}
		*/		
	}
}

/*
** Hue, saturation, luminance
*/

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
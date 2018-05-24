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

uniform sampler2D diffuseTexture;

void main( void )
{	
	gl_TexCoord[0] = gl_MultiTexCoord0; 
	
	vec3 v3Normal;		
		
	float fAngle;
	float fShininessFactor;
	
	// transform the vertext normal the same the object is transformed
	v3Normal   = gl_NormalMatrix * gl_Normal;

	// set normal length to 1.
	v3Normal   = normalize(v3Normal);

	// calculate the angle eye-position - vertex - light direction
	// the angle must not be less than 0.0
	fAngle = max(0.0, dot(v3Normal, vec3(gl_LightSource[0].halfVector)));
	
	// calculate the vertex shininess as the calculated angle powered to the materials shininess factor
	fShininessFactor = pow(fAngle, gl_FrontMaterial.shininess);
	
	vec4 diffuseC = texture2D( diffuseTexture, gl_TexCoord[0].st );
	// calculate the vertex color with the normal gouraud lighting calculation
	gl_FrontColor = gl_LightSource[0].ambient * gl_FrontMaterial.ambient +
       			    gl_LightSource[0].diffuse * gl_FrontMaterial.diffuse * fAngle +
		            gl_LightSource[0].specular * gl_FrontMaterial.specular * fShininessFactor;
	gl_FrontColor = diffuseC;

	// transform the vertex by the current opengl transformation matrix		
	gl_Position = ftransform();
}

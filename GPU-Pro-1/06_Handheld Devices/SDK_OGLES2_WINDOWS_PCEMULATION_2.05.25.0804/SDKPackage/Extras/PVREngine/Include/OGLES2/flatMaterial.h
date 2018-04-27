/******************************************************************************

 @File         flatMaterial.h

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  A default PFX to be used with the PVREngine when no other PFX is
               specified.

******************************************************************************/
#ifndef _FLAT_MATERIAL_H_
#define _FLAT_MATERIAL_H_

const char flatMaterial[] =
"[HEADER]\n"
"	VERSION		01.00.00.00\n"
"	DESCRIPTION Flat material\n"
"	COPYRIGHT	Img Tec\n"
"[/HEADER]\n"

"[EFFECT]\n"
"	NAME 	Flat\n"

	// GLOBALS UNIFORMS
"	UNIFORM myWVPMatrix 	WORLDVIEWPROJECTION\n"

	// ATTRIBUTES
"	ATTRIBUTE 	myVertex	POSITION\n"

"	VERTEXSHADER MyVertexShader\n"
"	FRAGMENTSHADER MyFragmentShader\n"

"[/EFFECT]\n"

"[VERTEXSHADER]\n"
"	NAME 		MyVertexShader\n"

	// LOAD GLSL AS CODE
"	[GLSL_CODE]\n"
"	attribute highp vec3	myVertex;\n"
"	uniform highp mat4	myWVPMatrix;\n"

"		void main(void)\n"
"		{\n"
"			gl_Position = myWVPMatrix * vec4(myVertex,1.0);\n"
"		}\n"
"	[/GLSL_CODE]\n"
"[/VERTEXSHADER]\n"

"[FRAGMENTSHADER]\n"
"	NAME 		MyFragmentShader\n"

	// LOAD GLSL AS CODE
"	[GLSL_CODE]\n"

"		void main (void)\n"
"		{\n"
"			gl_FragColor =  vec4(1.0,0.0,0.0,1.0);\n"
"		}\n"
"	[/GLSL_CODE]\n"
"[/FRAGMENTSHADER]\n";

#endif // _FLAT_MATERIAL_H_

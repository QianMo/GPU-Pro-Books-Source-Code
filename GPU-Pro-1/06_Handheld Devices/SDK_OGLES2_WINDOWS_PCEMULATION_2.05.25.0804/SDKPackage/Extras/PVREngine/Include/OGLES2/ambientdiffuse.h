/******************************************************************************

 @File         ambientdiffuse.h

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  A default PFX to be used with the PVREngine when no other PFX is
               specified.

******************************************************************************/
#ifndef _AMBIENT_DIFFUSE_H_
#define _AMBIENT_DIFFUSE_H_

const char ambdiffpointshader[] =
"[HEADER]\n"
"	VERSION		01.00.00.00\n"
"	DESCRIPTION Texturing with a single diffuse point light\n"
"	COPYRIGHT	Img Tec\n"
"[/HEADER]\n"

"[TEXTURES]\n"
"	FILE basemap		Flat.pvr	LINEAR-LINEAR-LINEAR\n"
"[/TEXTURES]\n"

"[EFFECT]\n"
"	NAME 	BasicDiffuse\n"

	// GLOBALS UNIFORMS
"	UNIFORM myViewIT 		WORLDIT\n"
"	UNIFORM myMVPMatrix 	WORLDVIEWPROJECTION\n"
"	UNIFORM	myLightDirection	LIGHTPOSWORLD\n"
"	UNIFORM mAmbient		MATERIALCOLORAMBIENT\n"
"	UNIFORM mDiffuse		MATERIALCOLORDIFFUSE\n"
"	UNIFORM	basemap			TEXTURE0\n"
"	UNIFORM fOpacity		MATERIALOPACITY\n"

	// ATTRIBUTES
"	ATTRIBUTE 	myVertex	POSITION\n"
"	ATTRIBUTE	myNormal	NORMAL\n"
"	ATTRIBUTE	myUV		UV\n"

"	VERTEXSHADER MyVertexShader\n"
"	FRAGMENTSHADER MyFragmentShader\n"
"	TEXTURE 0 basemap\n"

"[/EFFECT]\n"

"[VERTEXSHADER]\n"
"	NAME 		MyVertexShader\n"

	// LOAD GLSL AS CODE
"	[GLSL_CODE]\n"
"	attribute highp vec3	myVertex;\n"
"	attribute mediump vec3	myNormal;\n"
"	attribute mediump vec2	myUV;\n"

"	uniform highp mat4	myMVPMatrix;\n"
"	uniform mediump mat3	myViewIT;\n"
"	uniform mediump vec3	myLightDirection;\n"
"	uniform lowp	vec3	mAmbient;\n"
"	uniform lowp	vec3	mDiffuse;\n"

"	varying mediump vec3 	Normal;\n"
"	varying mediump vec2	texCoordinate;\n"
"	varying mediump vec3	DiffuseIntensity;\n"


"		void main(void)\n"
"		{\n"
"			gl_Position = myMVPMatrix * vec4(myVertex,1.0);\n"
"			Normal = normalize(myViewIT * myNormal);\n"
"			lowp float litFactor = dot(Normal, normalize(myLightDirection-myVertex));\n"
"			DiffuseIntensity = vec3(litFactor*mDiffuse)+mAmbient;\n"
"			texCoordinate = myUV.st;\n"
"		}\n"
"	[/GLSL_CODE]\n"
"[/VERTEXSHADER]\n"

"[FRAGMENTSHADER]\n"
"	NAME 		MyFragmentShader\n"

	// LOAD GLSL AS CODE
"	[GLSL_CODE]\n"
"		uniform sampler2D 		basemap;\n"
"		uniform lowp	float	fOpacity;\n"
"		varying mediump vec3	DiffuseIntensity;\n"
"		varying mediump vec2	texCoordinate;\n"

"		void main (void)\n"
"		{\n"
"			gl_FragColor =   texture2D(basemap, texCoordinate) * vec4(DiffuseIntensity,fOpacity);\n"
"		}\n"
"	[/GLSL_CODE]\n"
"[/FRAGMENTSHADER]\n";

#endif	// _AMBIENT_DIFFUSE_H_

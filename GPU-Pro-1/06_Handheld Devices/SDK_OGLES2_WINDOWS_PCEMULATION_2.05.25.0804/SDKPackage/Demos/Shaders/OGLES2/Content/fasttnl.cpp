// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: fasttnl.pfx ********

// File data
static const char _fasttnl_pfx[] = 
	"[HEADER]\r\n"
	"\tVERSION\t\t\t00.00.00.00\r\n"
	"\tDESCRIPTION\t\tFast Transformation and Lighting Example\r\n"
	"\tCOPYRIGHT\t\tImagination Technologies Ltd.\r\n"
	"[/HEADER]\r\n"
	"\r\n"
	"[TEXTURES]\r\n"
	"\tFILE base \tbase.pvr\t\tLINEAR-LINEAR-LINEAR\r\n"
	"[/TEXTURES]\r\n"
	"\r\n"
	"[VERTEXSHADER]\r\n"
	"\tNAME myVertShader\r\n"
	"\t[GLSL_CODE]\r\n"
	"\t\tattribute highp vec3\tmyVertex;\r\n"
	"\t\tattribute mediump vec3\tmyNormal;\r\n"
	"\t\tattribute mediump vec2\tmyUV;\r\n"
	"\t\tuniform mediump mat4\tmyMVPMatrix;\r\n"
	"\t\tuniform mediump mat3\tmyModelViewIT;\r\n"
	"\t\tconst mediump vec3\t\tmyLightDirection = vec3(0.6,0.6,0.6);\r\n"
	"\t\tconst mediump vec4\t\tmyMaterial = vec4(0.5,0.5,2.5,0.8);\r\n"
	"\t\tvarying lowp float\t\tDiffuseIntensity;\r\n"
	"\t\tvarying lowp float\t\tSpecularIntensity;\r\n"
	"\t\tvarying lowp vec2\t\ttexCoordinate;\r\n"
	"\t\tvoid main(void)\r\n"
	"\t\t{\r\n"
	"\t\t\t// Transform the position into clipping-space.\r\n"
	"\t\t\tgl_Position = myMVPMatrix * vec4(myVertex,1);\t\t\t\t\t\t\t\r\n"
	"\t\t\tmediump vec3 normal = normalize(myModelViewIT * myNormal);\r\n"
	"\t\t\tDiffuseIntensity = dot(normal, myLightDirection); \r\n"
	"\t\t\t// Substract and multiply DiffuseIntensity by Specular Bias (w) and Scale (z) to have a 'decent' looking specular effect\r\n"
	"\t\t\t// See code below for an explanation of how these parameters are calculated\r\n"
	"\t\t\tSpecularIntensity = max((DiffuseIntensity - myMaterial.w) * myMaterial.z, 0.0); \r\n"
	"\t\t\t// Pass the UV co-ordinates\r\n"
	"\t\t\ttexCoordinate = myUV.st;\r\n"
	"\t\t}\r\n"
	"\t[/GLSL_CODE]\r\n"
	"[/VERTEXSHADER]\r\n"
	"\r\n"
	"[FRAGMENTSHADER]\r\n"
	"\tNAME myFragShader\r\n"
	"\t[GLSL_CODE]\r\n"
	"\t\tuniform sampler2D sampler2d;\r\n"
	"\t\tvarying lowp float\tDiffuseIntensity;\r\n"
	"\t\tvarying lowp float\tSpecularIntensity;\r\n"
	"\t\tvarying lowp vec2\ttexCoordinate;\r\n"
	"\t\tvoid main (void)\r\n"
	"\t\t{\r\n"
	"\t\t\tlowp vec3 texColour  = vec3 (texture2D(sampler2d, texCoordinate));\r\n"
	"\t\t\tlowp vec3 finalColour = (texColour * DiffuseIntensity) + SpecularIntensity;\r\n"
	"\t\t\tgl_FragColor = vec4(finalColour, 1.0);\r\n"
	"\t\t}\r\n"
	"\t[/GLSL_CODE]\r\n"
	"[/FRAGMENTSHADER]\r\n"
	"\r\n"
	"[EFFECT]\r\n"
	"\tNAME myEffect\r\n"
	"\r\n"
	"\tATTRIBUTE\tmyVertex\t\t\tPOSITION\r\n"
	"\tATTRIBUTE\tmyNormal\t\t\tNORMAL\r\n"
	"\tATTRIBUTE\tmyUV\t\t\t\tUV\r\n"
	"\tUNIFORM\t\tmyMVPMatrix\t\t\tWORLDVIEWPROJECTION\r\n"
	"\tUNIFORM\t\tmyModelViewIT\t\tWORLDVIEWIT\r\n"
	"\tUNIFORM\t\tsampler2d\t\t\tTEXTURE0\r\n"
	"\r\n"
	"\tTEXTURE 0 base\r\n"
	"\r\n"
	"\tVERTEXSHADER myVertShader\r\n"
	"\tFRAGMENTSHADER myFragShader\r\n"
	"[/EFFECT]\r\n";

// Register fasttnl.pfx in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_fasttnl_pfx("fasttnl.pfx", _fasttnl_pfx, 2104);

// ******** End: fasttnl.pfx ********


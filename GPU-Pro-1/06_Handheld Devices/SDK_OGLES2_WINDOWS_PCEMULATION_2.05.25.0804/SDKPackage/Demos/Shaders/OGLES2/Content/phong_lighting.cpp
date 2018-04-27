// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: phong_lighting.pfx ********

// File data
static const char _phong_lighting_pfx[] = 
	"[HEADER]\r\n"
	"\tVERSION\t\t00.00.00.00\r\n"
	"\tDESCRIPTION Phong Lighting Example\r\n"
	"\tCOPYRIGHT\tImagination Technologies Ltd.\r\n"
	"[/HEADER]\r\n"
	"\r\n"
	"[TEXTURES]\r\n"
	"[/TEXTURES]\r\n"
	"\r\n"
	"[VERTEXSHADER]\r\n"
	"\tNAME myVertShader\r\n"
	"\t[GLSL_CODE]\r\n"
	"\t\tattribute highp vec4\tmyVertex;\r\n"
	"\t\tattribute mediump vec3\tmyNormal;\r\n"
	"\t\tuniform mediump mat4\tmyMVPMatrix;\r\n"
	"\t\tuniform mediump mat3\tmyModelViewIT;\r\n"
	"\t\tuniform mediump mat4\tmyModelView;\r\n"
	"\r\n"
	"\t\tconst mediump vec4\tLightSourcePosition = vec4(-1.0, 3.0, -2.0, 0.0);\r\n"
	"\t\tconst mediump vec4\tambient = vec4(0.2,0.2,0.2,1.0); \r\n"
	"\t\tconst mediump vec4\tambientGlobal = vec4(0.2,0.2,0.2,1.0);\r\n"
	"\t\tconst mediump vec4\tdiffuse = vec4(0.3,0.4,0.1,1.0); \r\n"
	"\t\t \r\n"
	"\t\tvarying mediump vec4 color; \r\n"
	"\r\n"
	"\t\tvoid main()\r\n"
	"\t\t{\r\n"
	"\t\t\tvec4 ecPos;\r\n"
	"\t\t\tmediump vec3 viewV,ldir;\r\n"
	"\t\t\tmediump float NdotL,NdotHV;\r\n"
	"\t\t\tmediump vec3 normal,lightDir,halfVector;\r\n"
	"\t\t\t\r\n"
	"\t\t\t// Transform the position\r\n"
	"\t\t\tgl_Position = myMVPMatrix * myVertex;\r\n"
	"\t\t\t\r\n"
	"\t\t\t// Transform the normal\r\n"
	"\t\t\tnormal = normalize(myModelViewIT * myNormal);\r\n"
	"\t\t\t\r\n"
	"\t\t\t// Compute the light's direction \r\n"
	"\t\t\tecPos = myModelView * myVertex;\r\n"
	"\t\t\tlightDir = normalize(vec3(LightSourcePosition-ecPos));\r\n"
	"\r\n"
	"\t\t\thalfVector = normalize(lightDir + vec3(ecPos));\r\n"
	"\t\t\t\t\r\n"
	"\t\t\t/* The ambient terms have been separated since one of them */\r\n"
	"\t\t\t/* suffers attenuation */\r\n"
	"\t\t\tcolor = ambientGlobal; \r\n"
	"\t\t\t\r\n"
	"\t\t\t/* compute the dot product between normal and normalized lightdir */\r\n"
	"\t\t\tNdotL = abs(dot(normal,lightDir));\r\n"
	"\r\n"
	"\t\t\tif (NdotL > 0.0) \r\n"
	"\t\t\t{\r\n"
	"\t\t\t\tcolor += (diffuse * NdotL + ambient);\t\r\n"
	"\t\t\t\t\r\n"
	"\t\t\t\tNdotHV = abs(dot(normal,halfVector));\r\n"
	"\t\t\t\tcolor += vec4(0.6,0.2,0.4,1.0) * pow(NdotHV,50.0);\r\n"
	"\t\t\t}\r\n"
	"\t\t}\r\n"
	"\t[/GLSL_CODE]\r\n"
	"[/VERTEXSHADER]\r\n"
	"\r\n"
	"[FRAGMENTSHADER]\r\n"
	"\tNAME myFragShader\r\n"
	"\t[GLSL_CODE]\r\n"
	"\r\n"
	"\tvarying mediump vec4 color;\r\n"
	"\r\n"
	"\t\tvoid main()\r\n"
	"\t\t{\r\n"
	"\t\t\tgl_FragColor = color;\r\n"
	"\t\t}\r\n"
	"\t[/GLSL_CODE]\r\n"
	"[/FRAGMENTSHADER]\r\n"
	"\r\n"
	"[EFFECT]\r\n"
	"\tNAME myEffect\r\n"
	"\r\n"
	"\tATTRIBUTE\tmyVertex\t\t\tPOSITION\r\n"
	"\tATTRIBUTE\tmyNormal\t\t\tNORMAL\r\n"
	"\tUNIFORM\t\tmyMVPMatrix\t\t\tWORLDVIEWPROJECTION\r\n"
	"\tUNIFORM\t\tmyModelView\t\t\tWORLDVIEW\r\n"
	"\tUNIFORM\t\tmyModelViewIT\t\tWORLDVIEWIT\r\n"
	"\r\n"
	"\tVERTEXSHADER myVertShader\r\n"
	"\tFRAGMENTSHADER myFragShader\r\n"
	"[/EFFECT]\r\n";

// Register phong_lighting.pfx in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_phong_lighting_pfx("phong_lighting.pfx", _phong_lighting_pfx, 2088);

// ******** End: phong_lighting.pfx ********


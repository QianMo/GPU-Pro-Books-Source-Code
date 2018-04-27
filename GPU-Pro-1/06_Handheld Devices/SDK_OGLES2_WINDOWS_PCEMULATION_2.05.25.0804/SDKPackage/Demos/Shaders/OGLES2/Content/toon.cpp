// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: toon.pfx ********

// File data
static const char _toon_pfx[] = 
	"[HEADER]\r\n"
	"\tVERSION\t\t00.00.00.00\r\n"
	"\tDESCRIPTION Toon Example\r\n"
	"\tCOPYRIGHT\tImagination Technologies Ltd.\r\n"
	"[/HEADER]\r\n"
	"\r\n"
	"[TEXTURES]\r\n"
	"[/TEXTURES]\r\n"
	"\r\n"
	"[VERTEXSHADER]\r\n"
	"\tNAME myVertShader\r\n"
	"\t[GLSL_CODE]\r\n"
	"\t\tattribute mediump vec4\tmyVertex;\r\n"
	"\t\tattribute mediump vec3\tmyNormal;\r\n"
	"\t\tattribute mediump vec4\tmyUV;\r\n"
	"\t\tuniform mediump mat4\tmyMVPMatrix;\r\n"
	"\t\tuniform mediump mat3\tmyModelViewIT;\r\n"
	"\t\tvarying mediump float\tintensity;\r\n"
	"\r\n"
	"\t\tvoid main(void)\r\n"
	"\t\t{\r\n"
	"\t\t\tmediump vec3 Normal;\r\n"
	"\t\t\tgl_Position = myMVPMatrix * myVertex;\r\n"
	"\t\t\tNormal = myModelViewIT * myNormal;\r\n"
	"\t\t\tintensity = abs( dot(vec3(0,0,1),normalize(Normal)) );\r\n"
	"\t\t}\r\n"
	"\t[/GLSL_CODE]\r\n"
	"[/VERTEXSHADER]\r\n"
	"\r\n"
	"[FRAGMENTSHADER]\r\n"
	"\tNAME myFragShader\r\n"
	"\t[GLSL_CODE]\r\n"
	"\t\tvarying mediump float intensity;\r\n"
	"\t\t\r\n"
	"\t\tvoid main (void)\r\n"
	"\t\t{\r\n"
	"\t\t\tmediump vec4 color;\t\r\n"
	"\t\t\tif (intensity > 0.95)\r\n"
	"\t\t\t\tcolor = vec4(1.0,0.5,0.5,1.0);\r\n"
	"\t\t\telse if (intensity > 0.5)\r\n"
	"\t\t\t\tcolor = vec4(0.6,0.3,0.3,1.0);\r\n"
	"\t\t\telse if (intensity > 0.25)\r\n"
	"\t\t\t\tcolor = vec4(0.4,0.2,0.2,1.0);\r\n"
	"\t\t\telse\r\n"
	"\t\t\t\tcolor = vec4(0.2,0.1,0.1,1.0);\r\n"
	" \t\t\tgl_FragColor = color;\r\n"
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
	"\tUNIFORM\t\tmyModelViewIT\t\tWORLDVIEWIT\r\n"
	"\r\n"
	"\tVERTEXSHADER myVertShader\r\n"
	"\tFRAGMENTSHADER myFragShader\r\n"
	"[/EFFECT]\r\n";

// Register toon.pfx in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_toon_pfx("toon.pfx", _toon_pfx, 1332);

// ******** End: toon.pfx ********


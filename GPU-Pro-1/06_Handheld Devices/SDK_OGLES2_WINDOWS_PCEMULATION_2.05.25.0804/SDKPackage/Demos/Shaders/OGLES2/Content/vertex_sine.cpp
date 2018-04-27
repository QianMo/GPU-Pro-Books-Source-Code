// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: vertex_sine.pfx ********

// File data
static const char _vertex_sine_pfx[] = 
	"[HEADER]\r\n"
	"\tVERSION\t\t00.00.00.00\r\n"
	"\tDESCRIPTION \r\n"
	"\tCOPYRIGHT\tImagination Technologies Ltd.\r\n"
	"[/HEADER]\r\n"
	"\r\n"
	"[VERTEXSHADER] \r\n"
	"\tNAME \t\tMyVertexShader \r\n"
	"\r\n"
	"\t[GLSL_CODE]\r\n"
	"\r\n"
	"attribute mediump vec4\tmyVertex;\r\n"
	"attribute mediump vec3\tmyNormal;\r\n"
	"uniform mediump mat4\tmyWVPMatrix;\r\n"
	"uniform mediump float\tmyAnim;\r\n"
	"uniform mediump mat3\tmyWorldViewIT;\r\n"
	"const vec3 LightPosition = vec3(0.0,4.0,0.0);\r\n"
	"const vec3 SurfaceColor = vec3(0.7, 0.8, 0.4);\r\n"
	"const float scaleIn = 1.0;\r\n"
	"const float scaleOut = 0.1;\r\n"
	"varying highp vec4 Color;\r\n"
	"\r\n"
	"void main(void)\r\n"
	"{\r\n"
	"   \r\n"
	"\tvec3 normal = myNormal; \r\n"
	"\r\n"
	"\tfloat ripple = 3.0*cos(0.2*myVertex.y + (radians(5.0*myAnim*360.0)));\r\n"
	"\tfloat ripple2 = -0.5*sin(0.2*myVertex.y + (radians(5.0*myAnim*360.0)));\r\n"
	"\t\r\n"
	"\tvec3 vertex = myVertex.xyz + vec3(0,0.0, ripple);\r\n"
	"    gl_Position = myWVPMatrix * vec4(vertex,1.0);\r\n"
	"\r\n"
	"\tnormal = normalize(myWorldViewIT * (myNormal + vec3(0,0.0, ripple2)) );\r\n"
	"\t\r\n"
	"\tvec3 position = vec3(myWVPMatrix * vec4(vertex,1.0));\r\n"
	"    \tvec3 lightVec   = vec3(0.0,0.0,1.0);\r\n"
	"    \r\n"
	"    float diffuse   = max(dot(lightVec, normal), 0.0);\r\n"
	"\r\n"
	"    if (diffuse < 0.125)\r\n"
	"         diffuse = 0.125;\r\n"
	"         \r\n"
	"    Color = vec4(SurfaceColor * diffuse * 1.5, 1.0);\r\n"
	" }\r\n"
	"\r\n"
	"\t[/GLSL_CODE]\r\n"
	"[/VERTEXSHADER]\r\n"
	"    \r\n"
	"[FRAGMENTSHADER] \r\n"
	"\tNAME \t\tMyFragmentShader \r\n"
	"\r\n"
	"\t[GLSL_CODE]\r\n"
	"varying highp vec4 Color;\r\n"
	"\r\n"
	"void main (void)\r\n"
	"{\r\n"
	"    gl_FragColor = Color;\r\n"
	"}\r\n"
	"\t[/GLSL_CODE]\r\n"
	"[/FRAGMENTSHADER]\r\n"
	" \r\n"
	"[EFFECT] \r\n"
	"\tNAME \tmyEffect\r\n"
	"\tATTRIBUTE\tmyVertex\t\tPOSITION\r\n"
	"\tATTRIBUTE\tmyNormal\t\tNORMAL\r\n"
	"\tUNIFORM\t\tmyAnim\t\t\tANIMATION\r\n"
	"\tUNIFORM\t\tmyWorldViewIT\tWORLDVIEWIT\t\r\n"
	"\tUNIFORM\t\tmyWVPMatrix\t\tWORLDVIEWPROJECTION\r\n"
	"\t\r\n"
	"\tVERTEXSHADER MyVertexShader\r\n"
	"\tFRAGMENTSHADER MyFragmentShader\r\n"
	"[/EFFECT]\r\n";

// Register vertex_sine.pfx in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_vertex_sine_pfx("vertex_sine.pfx", _vertex_sine_pfx, 1682);

// ******** End: vertex_sine.pfx ********


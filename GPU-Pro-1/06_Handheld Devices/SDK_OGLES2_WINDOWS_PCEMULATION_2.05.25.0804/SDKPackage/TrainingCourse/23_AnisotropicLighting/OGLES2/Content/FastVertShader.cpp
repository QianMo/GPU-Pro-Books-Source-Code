// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: FastVertShader.vsh ********

// File data
static const char _FastVertShader_vsh[] = 
	"/******************************************************************************\r\n"
	"* Vertex Shader (Fast method)\r\n"
	"*******************************************************************************\r\n"
	" This technique uses the dot product between the light direction and the normal\r\n"
	" to generate an x coordinate. The dot product between the half angle vector \r\n"
	" (vector half way between the viewer's eye and the light direction) and the \r\n"
	" normal to generate a y coordinate. These coordinates are used to lookup the \r\n"
	" intensity of light from the special image, which is accessible to the shader \r\n"
	" as a 2d texture. The intensity is then used to shade a fragment and hence \r\n"
	" create an anisotropic lighting effect.\r\n"
	"******************************************************************************/\r\n"
	"\r\n"
	"attribute highp vec3  inVertex;\r\n"
	"attribute highp vec3  inNormal;\r\n"
	"\r\n"
	"uniform highp mat4  MVPMatrix;\r\n"
	"uniform highp vec3  msLightDir;\r\n"
	"uniform highp vec3  msEyePos;\r\n"
	"\r\n"
	"varying mediump vec2  TexCoord;\r\n"
	"\r\n"
	"void main() \r\n"
	"{ \r\n"
	"\t// transform position\r\n"
	"\tgl_Position = MVPMatrix * vec4(inVertex, 1);\r\n"
	"\t\r\n"
	"\t// Calculate eye direction in model space\r\n"
	"\thighp vec3 msEyeDir = normalize(msEyePos - inVertex);\r\n"
	"\t\r\n"
	"\t// Calculate vector half way between the vertexToEye and light directions.\r\n"
	"\t// (division by 2 ignored as it is irrelevant after normalisation)\r\n"
	"\thighp vec3 halfAngle = normalize(msEyeDir + msLightDir); \r\n"
	"\t\r\n"
	"\t// Use dot product of light direction and normal to generate s coordinate.\r\n"
	"\t// We use GL_CLAMP_TO_EDGE as texture wrap mode to clamp to 0 \r\n"
	"\tTexCoord.s = dot(msLightDir, inNormal); \r\n"
	"\t// Use dot product of half angle and normal to generate t coordinate.\r\n"
	"\tTexCoord.t = dot(halfAngle, inNormal); \r\n"
	"} \r\n";

// Register FastVertShader.vsh in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FastVertShader_vsh("FastVertShader.vsh", _FastVertShader_vsh, 1698);

// ******** End: FastVertShader.vsh ********

// This file was created by Filewrap 1.1
// Little endian mode
// DO NOT EDIT

#include "../PVRTMemoryFileSystem.h"

// using 32 bit to guarantee alignment.
#ifndef A32BIT
 #define A32BIT static const unsigned int
#endif

// ******** Start: FastVertShader.vsc ********

// File data
A32BIT _FastVertShader_vsc[] = {
0x10fab438,0x685c78dd,0x30050100,0x2101,0xa9142cc2,0x0,0x0,0x92030000,0x0,0x4000000,0x0,0x9000000,0x2,0x0,0x20000,0x0,0x0,0x4a020000,0x55535020,0x17,0x23e,0x1,0x0,0x80c,0x0,0x2,0x79,0x0,0x8,0x0,0xffffffff,0x0,0x76000a,0xffff,0x50007,0x0,0x160000,0x0,0x0,0x0,0x0,0xfffc0000,0x0,0x0,0x0,0x20000,0xffffffff,0x160000,0x10007,0x14,0x10000,0x10015,0x10000,0x20016,0x10000,0x30001,0x10000,0x40002,0x10000,0x50003,0x10000,0x60004,0x10000,0x70005,
0x10000,0x80006,0x10000,0x90007,0x10000,0xa0008,0x10000,0xb0009,0x10000,0xc000a,0x10000,0xd000b,0x10000,0xe000c,0x10000,0xf000d,0x10000,0x10000e,0x10000,0x11000f,0x10000,0x120010,0x10000,0x130011,0x10000,0x140012,0x10000,0x150013,0x770000,0x40000,0x20000,0x2,0x19,0x80018001,0x80018001,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x10001,0x1,0x10001,0x10001,0x10001,0x1a0d0001,0x10847020,0x5a0e0082,0x10847060,0x9a0f0082,0x10867060,0x41830082,0x14000000,0xc1800080,0x10048000,0x80,0x12000000,0x400a0880,0x10003040,0xc00b0080,
0x10003080,0xc00c0080,0x10063060,0x82040080,0x14000020,0xc1810080,0x10048020,0x800080,0x12000020,0xc0b00880,0x10041000,0x9130081,0x606f060,0x4b1738ab,0x606f060,0x8d1b3882,0x606f060,0x1a1e3882,0x1001705f,0x410400a2,0x6020060,0x57c38ab,0x606c061,0x45833898,0x606e061,0x86003882,0x205c081,0x81138a2,0x605f0a0,0x4a153898,0x606f000,0x8c193882,0x606f000,0x1a1c3882,0x1001701f,0x600a2,0x18000000,0x803f,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x6e690700,0x74726556,0x7865,
0x40400,0x100,0x4000001,0x700,0x505f6c67,0x7469736f,0x6e6f69,0x5050100,0x10000,0x100,0xf0004,0x50564d00,0x7274614d,0x7869,0x31600,0x100,0x10040001,0xffff,0x7945736d,0x736f5065,0x4000000,0x1000003,0x10000,0x7000301,0x736d0000,0x6867694c,0x72694474,0x4000000,0x1000003,0x10000,0x7000314,0x6e690000,0x6d726f4e,0x6c61,0x40400,0x100,0x4040001,0x700,0x43786554,0x64726f6f,0x3000000,0x1000005,0x10000,0x3000200,0x0,
};

// Register FastVertShader.vsc in memory file system at application startup time
static CPVRTMemoryFileSystem RegisterFile_FastVertShader_vsc("FastVertShader.vsc", _FastVertShader_vsc, 946);

// ******** End: FastVertShader.vsc ********


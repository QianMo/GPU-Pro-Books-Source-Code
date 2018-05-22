#version 120
#extension GL_EXT_gpu_shader4 : require 

void main()
{	
	gl_Position = ftransform();
	gl_TexCoord[0] = gl_MultiTexCoord0;
}

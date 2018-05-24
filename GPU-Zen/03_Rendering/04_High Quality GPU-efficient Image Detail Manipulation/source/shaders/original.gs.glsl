#version 430 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out vec2 texcoord;

void main(void)
{

	//	Upper-Right
	gl_Position = vec4( 0.0, 1.0, 0.5, 1.0 );
    texcoord = vec2( 0.89, 0.79 );
    EmitVertex();

	//	Upper-Left
    gl_Position = vec4(-1.0, 1.0, 0.5, 1.0 );
    texcoord = vec2( 0.11, 0.79 ); 
    EmitVertex();

	//	Lower-Right
    gl_Position = vec4( 0.0,-0.33, 0.5, 1.0 );
    texcoord = vec2( 0.89, 0.21 ); 
    EmitVertex();

	//	Lower-Left
    gl_Position = vec4(-1.0,-0.33, 0.5, 1.0 );
    texcoord = vec2( 0.11, 0.21 ); 
    EmitVertex();

    EndPrimitive(); 

}
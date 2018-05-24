#version 430 core

layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

out vec2 texCoord_0;
out vec2 texCoord_1;
out vec2 texCoord_2;

void main(void)
{

/*	ORIGINAL
	//	Upper-Right
	gl_Position = vec4( -0.5, -0.33, 0.5, 1.0 );
    texCoord = vec2( 0.89, 0.79 );
    EmitVertex();

	//	Upper-Left
    gl_Position = vec4(-1.0, -0.33, 0.5, 1.0 );
    texCoord = vec2( 0.11, 0.79 ); 
    EmitVertex();

	//	Lower-Right
    gl_Position = vec4(-0.5,-1.0, 0.5, 1.0 );
    texCoord = vec2( 0.89, 0.21 ); 
    EmitVertex();

	//	Lower-Left
    gl_Position = vec4(-1.0,-1.0, 0.5, 1.0 );
    texCoord = vec2( 0.11, 0.21 ); 
    EmitVertex();
*/

	//	0-upper
    gl_Position = vec4(-1.0, -0.33, 0.5, 1.0 );
    texCoord_0 = vec2( 0.11, 0.79 ); 
    texCoord_1 = vec2( 0.00, 0.79 ); 
    texCoord_2 = vec2( 0.00, 0.79 ); 
    EmitVertex();

	//	0-lower
    gl_Position = vec4(-1.0, -1.0, 0.5, 1.0 );
    texCoord_0 = vec2( 0.11, 0.21 ); 
    texCoord_1 = vec2( 0.00, 0.21 ); 
    texCoord_2 = vec2( 0.00, 0.21 ); 
    EmitVertex();

	//	1-upper
	gl_Position = vec4( -0.5, -0.33, 0.5, 1.0 );
    texCoord_0 = vec2( 0.89, 0.79 );
    texCoord_1 = vec2( 0.11, 0.79 ); 
    texCoord_2 = vec2( 0.00, 0.79 ); 
    EmitVertex();

	//	1-lower
    gl_Position = vec4( -0.5, -1.0, 0.5, 1.0 );
    texCoord_0 = vec2( 0.89, 0.21 ); 
    texCoord_1 = vec2( 0.11, 0.21 ); 
    texCoord_2 = vec2( 0.00, 0.21 ); 
    EmitVertex();

	//	2-upper
	gl_Position = vec4( 0.0, -0.33, 0.5, 1.0 );
    texCoord_0 = vec2( 1.00, 0.79 );
    texCoord_1 = vec2( 0.89, 0.79 );
    texCoord_2 = vec2( 0.11, 0.79 );
    EmitVertex();

	//	2-lower
    gl_Position = vec4( 0.0,-1.0, 0.5, 1.0 );
    texCoord_0 = vec2( 1.00, 0.21 ); 
    texCoord_1 = vec2( 0.89, 0.21 ); 
    texCoord_2 = vec2( 0.11, 0.21 ); 
    EmitVertex();

	//	3-upper
	gl_Position = vec4( 0.5, -0.33, 0.5, 1.0 );
    texCoord_0 = vec2( 1.00, 0.79 ); 
    texCoord_1 = vec2( 1.00, 0.79 ); 
    texCoord_2 = vec2( 0.89, 0.79 );
    EmitVertex();

	//	3-lower
    gl_Position = vec4( 0.5, -1.0, 0.5, 1.0 );
    texCoord_0 = vec2( 1.00, 0.21 ); 
    texCoord_1 = vec2( 1.00, 0.21 ); 
    texCoord_2 = vec2( 0.89, 0.21 ); 
    EmitVertex();

    EndPrimitive(); 

}
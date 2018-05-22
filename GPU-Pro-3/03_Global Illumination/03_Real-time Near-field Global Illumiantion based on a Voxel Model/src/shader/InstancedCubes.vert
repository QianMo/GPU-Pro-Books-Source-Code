#extension GL_EXT_gpu_shader4 : require
#extension GL_ARB_draw_instanced : require
 
uniform samplerBuffer tboTranslate;

uniform vec3 scale;
 
void main()
{
	vec3 translate = texelFetchBuffer(tboTranslate, gl_InstanceID).rgb;
	vec4 vertex = vec4((gl_Vertex.xyz * scale + translate), 1.0);
 
	gl_Position = gl_ModelViewProjectionMatrix * vertex;
   gl_FrontColor = gl_Color;
 }


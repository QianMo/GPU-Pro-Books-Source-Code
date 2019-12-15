#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform float _SplitScreenMode;
uniform float _AspectRatio;

out vec2 uv;

const vec2 quadVertices[4] = vec2[4]( vec2( -1.0, -1.0), vec2( 1.0, -1.0), vec2( -1.0, 1.0), vec2( 1.0, 1.0));
const vec2 quadUVs[4] = vec2[4]( vec2( -1.0, 1.0), vec2( 1.0, 1.0), vec2( -1.0, -1.0), vec2( 1.0, -1.0));
void main()
{
	// Display quad only on left hand side of the screen in split screen mode
	vec4 pos = vec4(quadVertices[gl_VertexID], 0.0f, 1.0f);
	pos.x = clamp(pos.x, -1.0f, 1.0f * (1.0f - _SplitScreenMode));
	vec2 texcoord = quadUVs[gl_VertexID];
	texcoord.x = clamp(texcoord.x, -1.0f, 1.0f * (1.0f - _SplitScreenMode));

	// Account for window aspect ratio
	texcoord.x *= _AspectRatio;

	gl_Position = pos;
	uv = texcoord;
}

#version 330 core

uniform sampler2D input;
uniform vec4 _ScaleTranslate;
uniform float _SplitScreenMode;

in vec2 uv;
out vec4 color;

void main()
{
	color = texture(input, uv * _ScaleTranslate.xy + _ScaleTranslate.zw);

	// Display black separation line in split screen mode
	color *= float(!(_SplitScreenMode > 0.5f && uv.x > -0.006f));
}

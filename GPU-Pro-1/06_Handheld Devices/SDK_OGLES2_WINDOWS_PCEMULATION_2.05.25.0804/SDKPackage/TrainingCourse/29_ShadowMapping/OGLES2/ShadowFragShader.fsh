uniform highp sampler2D sShadow;
uniform sampler2D sTexture;

varying highp vec4 projCoord;
varying mediump vec2 texCoord;
varying lowp vec3 LightIntensity;

void main ()
{	
	// Subtract a small magic number to account for floating-point error
	highp float comp = (projCoord.z / projCoord.w) - 0.03;
	highp float depth = texture2DProj(sShadow, projCoord).r;
	
	lowp float val = comp <= depth ? 1.0 : 0.4;
	lowp vec3 color = texture2D(sTexture, texCoord).rgb * LightIntensity * val;
	
	gl_FragColor = vec4(color, 1.0);
}


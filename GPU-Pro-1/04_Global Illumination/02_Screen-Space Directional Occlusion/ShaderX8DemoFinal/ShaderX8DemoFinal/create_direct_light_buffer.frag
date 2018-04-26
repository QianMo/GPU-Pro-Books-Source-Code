
varying vec4 position;
varying vec3 normal;

uniform vec4 lightPosition;
uniform vec4 lightColor;

uniform sampler2D positionTexture;
uniform sampler2D normalTexture;
uniform sampler2D reflectanceTexture;


void main()
{	
	vec4 pixelPosition = texelFetch2D(positionTexture, ivec2(gl_FragCoord.xy), 0);
	vec4 pixelNormal = texelFetch2D(normalTexture, ivec2(gl_FragCoord.xy), 0);
	vec4 pixelReflectance = texelFetch2D(reflectanceTexture, ivec2(gl_FragCoord.xy), 0);
	
	vec3 lightVector = lightPosition.xyz - pixelPosition.xyz;       // vector to light source
	float lightDistance = length(lightVector);         // distance to light source
	vec3 lightDir = lightVector / lightDistance;       // normalized vector to light source

	pixelNormal.w = 0.0;
	pixelNormal = normalize(pixelNormal);
	float cosAlpha = max(0.0, dot(lightDir.xyz, pixelNormal.xyz));
	
	vec4 diffuseColor = vec4(cosAlpha) * pixelReflectance;
	// vec4 ambientColor = vec4(0.2, 0.2, 0.2, 1.0);
	
	// diffuseColor += ambientColor;

	gl_FragColor = diffuseColor;
	gl_FragColor.a = 1.0;
}

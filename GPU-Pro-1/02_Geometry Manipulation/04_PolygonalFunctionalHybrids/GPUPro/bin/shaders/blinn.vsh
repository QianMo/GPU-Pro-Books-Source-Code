varying	vec3 lightOut;
varying	vec3 halfOut;
varying	vec3 normalOut;
varying vec3 viewerOut;
varying vec4 colorOut;
varying vec2 uvOut;

attribute vec4 colorCustom;
attribute vec4 jointIndices;
attribute vec4 jointWeights;
attribute vec3 normal;
attribute vec2 uvCoords;

uniform	vec4	lightPosition;
uniform	vec4	eyePosition;

uniform	mat4	jointMatrices[50];

void main(void)
{
	vec4 position = ((gl_Vertex * jointMatrices[int(jointIndices.x)]) * jointWeights.x);
	
	position += ((gl_Vertex * jointMatrices[int(jointIndices.y)]) * jointWeights.y);
	position += ((gl_Vertex * jointMatrices[int(jointIndices.z)]) * jointWeights.z);
	position += ((gl_Vertex * jointMatrices[int(jointIndices.w)]) * jointWeights.w);
	
	// for simplicity, it can be assumed that there's not non-uniform scaling in the joints
	vec4 curNormal = (vec4(gl_Normal, 0) * jointMatrices[int(jointIndices.x)]) * jointWeights.x;
	
	curNormal += (vec4(gl_Normal, 0) * jointMatrices[int(jointIndices.y)]) * jointWeights.y;
	curNormal += (vec4(gl_Normal, 0) * jointMatrices[int(jointIndices.z)]) * jointWeights.z;
	curNormal += (vec4(gl_Normal, 0) * jointMatrices[int(jointIndices.w)]) * jointWeights.w;
	
	vec4	p = gl_ModelViewMatrix * (position);

	lightOut = normalize (  lightPosition - p ).xyz;
	viewerOut = normalize ( eyePosition  - p ).xyz;
	halfOut = normalize ( lightOut + viewerOut );

	normalOut = normalize ( gl_NormalMatrix * curNormal.xyz );
	colorOut = colorCustom;
	
	uvOut = uvCoords;

	
	gl_Position = gl_ModelViewProjectionMatrix * (position);
}

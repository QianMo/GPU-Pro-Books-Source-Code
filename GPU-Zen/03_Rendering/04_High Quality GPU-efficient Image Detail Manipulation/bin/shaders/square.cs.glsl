#version 430 core

layout(local_size_x = 1024) in;
layout(binding = 0, rgba32f)  readonly uniform image2D inImage;
layout(binding = 1, rgba32f) writeonly uniform image2D outImage;

shared vec3 shData[gl_WorkGroupSize.x * 2];

void main() {

	uint	id = gl_LocalInvocationID.x;
	ivec2	texCoord = ivec2(id, gl_WorkGroupID.x);

	//	Co-operative read in
	vec3 pixel = imageLoad(inImage, texCoord).rgb;

	pixel = pixel * pixel;

	//	Co-operative write back
	imageStore(outImage, texCoord, vec4(pixel.rgb, 1.0));

}

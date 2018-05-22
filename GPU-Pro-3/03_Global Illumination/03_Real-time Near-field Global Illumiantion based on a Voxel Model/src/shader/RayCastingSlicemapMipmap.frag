#version 120
#extension GL_EXT_gpu_shader4 : require

uniform sampler2D texPositionRayStart;
uniform sampler2D texPositionRayEnd;

uniform usampler2D voxelTexture; // level 0: width x height, ..., level n: 1x1
uniform usampler1D bitmask; 
uniform int level;

uniform mat4 viewMatrixVoxelCam;
uniform mat4 projMatrixVoxelCam;
uniform mat4 inverseViewMatrixUserCam;

bool intersectBits(in uvec4 bitRay, in vec2 texel)
{
	uvec4 result = bitRay & texture2DLod(voxelTexture, texel, level);		
	return (result != uvec4(0));
}

vec3 eyeToWindow(in vec3 p)
{
	vec4 clipCoord = projMatrixVoxelCam * vec4(p, 1.0);
	vec3 nDeviceCoord = clipCoord.xyz / clipCoord.w; //[-1,1 ..-1,1 .. -1,1 ]
	vec3 lookupCoord = (nDeviceCoord+1.0)/2.0;
	return lookupCoord;
}


void main()
{
	gl_FragColor = vec4(1.0);

	vec4 rayStart = texture2D(texPositionRayStart, gl_TexCoord[0].st);

	if(rayStart.w != 0.0) // w == 1.0: positon written to texture
   {
		vec4 rayEnd = texture2D(texPositionRayEnd, gl_TexCoord[0].st);

		rayStart = viewMatrixVoxelCam * inverseViewMatrixUserCam * rayStart;
		rayEnd   = viewMatrixVoxelCam * inverseViewMatrixUserCam * rayEnd;

		vec3 rayDir = rayEnd.xyz - rayStart.xyz;

      int steps = max(64, textureSize2D(voxelTexture, level).x * 2);
		float inc = 1.0 / float(steps);
      float t = 0.0;
		for(t = 0.0f; t < 1.0f; t += inc)
      {
			vec3 posOnRay = rayStart.xyz + (t * rayDir);
			vec3 viewPortPos = eyeToWindow(posOnRay);

			if(intersectBits(texture1D(bitmask, viewPortPos.z), viewPortPos.st))
         {
				gl_FragColor = vec4(viewPortPos, 1);
				break;
			}
		}
	}


}
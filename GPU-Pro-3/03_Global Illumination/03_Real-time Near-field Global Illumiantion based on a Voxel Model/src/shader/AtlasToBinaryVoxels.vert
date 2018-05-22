///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#version 120
#extension GL_EXT_gpu_shader4: require

uniform sampler2D textureAtlas; // contains positions in world space

uniform mat4 viewProjMatrixVoxelCam;

varying float mappedZ;

void main ()
{ 
   // Incoming vertices have positions in the range of
   // [0..atlasWidth-1]x[0..atlasHeight-1].
   // Fetch world space position from atlas.
   vec3 pos3D = texelFetch2D(textureAtlas, ivec2(gl_Vertex.xy), 0).rgb;

   // Transform into voxel grid coordinates
   gl_Position = viewProjMatrixVoxelCam * vec4(pos3D, 1.0); 
  
   // Map z-coordinate to [0,1]
   mappedZ = gl_Position.z * 0.5 + 0.5;   
}
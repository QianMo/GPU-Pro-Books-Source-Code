SB_LAYOUT(CUSTOM1_SB_BP) buffer LightBuffer 
{
  TiledShadowLight lights[];
} lightBuffer;

in gl_PerVertex
{
  vec4 gl_Position;
} gl_in[3]; 

in VS_Output
{
  uint lightIndex;
} inputGS[3];

const vec3 planeNormals[12] =
{
  vec3(0.00000000, -0.03477280, 0.99939519),
  vec3(-0.47510946, -0.70667917, 0.52428567),
  vec3(0.47510946, -0.70667917, 0.52428567),
  vec3(0.00000000, -0.03477280, -0.99939519),
  vec3(0.47510946, -0.70667917, -0.52428567),
  vec3(-0.47510946, -0.70667917, -0.52428567),
  vec3(-0.52428567, 0.70667917, -0.47510946),
  vec3(-0.52428567, 0.70667917, 0.47510946),
  vec3(-0.99939519, 0.03477280, 0.00000000),
  vec3(0.52428567, 0.70667917, -0.47510946),
  vec3(0.99939519, 0.03477280, 0.00000000),
  vec3(0.52428567, 0.70667917, 0.47510946)
};

float GetClipDistance(in vec3 lightPosition, in uint vertexIndex, in uint planeIndex)
{
  vec3 normal = planeNormals[planeIndex];
  return (dot(gl_in[vertexIndex].gl_Position.xyz, normal)+dot(-normal, lightPosition));
}

layout(triangles) in;
layout(triangle_strip, max_vertices = 12) out;
void main()
{
  const uint lightIndex = inputGS[0].lightIndex;
  const vec3 lightPosition = lightBuffer.lights[lightIndex].position;

  // back-face culling
  vec3 normal = cross(gl_in[2].gl_Position.xyz-gl_in[0].gl_Position.xyz, gl_in[0].gl_Position.xyz - gl_in[1].gl_Position.xyz);
  vec3 view = lightPosition-gl_in[0].gl_Position.xyz;
  if(dot(normal, view) < 0.0f)
    return;

  for(uint faceIndex=0; faceIndex<4; faceIndex++)
  {
    uint inside = 0;
    float clipDistances[9];
    for(uint sideIndex=0; sideIndex<3; sideIndex++)
    {
      const uint planeIndex = (faceIndex*3)+sideIndex;
      const uint bit = 1 << sideIndex;
      for(uint vertexIndex=0; vertexIndex<3; vertexIndex++)
      {
        uint clipDistanceIndex = sideIndex*3+vertexIndex;
        clipDistances[clipDistanceIndex] = GetClipDistance(lightPosition, vertexIndex, planeIndex);
        inside |= (clipDistances[clipDistanceIndex] > 0.001) ? bit : 0;
      }
    }

    if(inside == 0x7)
    {
      const mat4 shadowViewProjTexMatrix = lightBuffer.lights[lightIndex].shadowViewProjTexMatrices[faceIndex];
      for(uint vertexIndex=0; vertexIndex<3; vertexIndex++)
      {
        gl_Position = shadowViewProjTexMatrix*gl_in[vertexIndex].gl_Position;
        gl_ClipDistance[0] = clipDistances[vertexIndex];
        gl_ClipDistance[1] = clipDistances[3+vertexIndex];
        gl_ClipDistance[2] = clipDistances[6+vertexIndex];
        EmitVertex();  
      }
      EndPrimitive(); 
    }
  }
}


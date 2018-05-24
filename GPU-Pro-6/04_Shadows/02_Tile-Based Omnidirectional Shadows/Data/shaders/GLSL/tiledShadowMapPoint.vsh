SB_LAYOUT(CUSTOM0_SB_BP) buffer ShadowLightIndexBuffer 
{
  uint counter;
  uint lightIndices[];
} shadowLightIndexBuffer;

layout(location = POSITION_ATTRIB) in vec3 inputPosition; 

out gl_PerVertex 
{
  vec4 gl_Position;
};

out VS_Output
{
  uint lightIndex;
} outputVS;

void main()
{
  gl_Position = vec4(inputPosition, 1.0);
  uint offset = gl_BaseInstanceARB+gl_InstanceID;
  outputVS.lightIndex = shadowLightIndexBuffer.lightIndices[offset];
}


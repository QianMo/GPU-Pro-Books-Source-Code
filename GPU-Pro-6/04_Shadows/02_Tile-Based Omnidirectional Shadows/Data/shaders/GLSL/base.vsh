GLOBAL_CAMERA_UB(cameraUB);

layout(location = POSITION_ATTRIB) in vec3 inputPosition; 
layout(location = TEXCOORDS_ATTRIB) in vec2 inputTexCoords; 
layout(location = NORMAL_ATTRIB) in vec3 inputNormal; 
layout(location = TANGENT_ATTRIB) in vec4 inputTangent; 

out VS_Output
{
  vec2 texCoords;
  vec3 normal;
  vec3 tangent;
  vec3 bitangent;
} outputVS;

void main()
{
  vec4 positionWS = vec4(inputPosition, 1.0f);
  gl_Position = cameraUB.viewProjMatrix*positionWS;
  outputVS.texCoords = inputTexCoords;
        
  outputVS.normal = inputNormal;
  outputVS.tangent = inputTangent.xyz;
  outputVS.bitangent = cross(inputNormal, inputTangent.xyz)*inputTangent.w; 
}


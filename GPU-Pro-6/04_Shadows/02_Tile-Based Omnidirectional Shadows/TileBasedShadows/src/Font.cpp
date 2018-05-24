#include <stdafx.h>
#include <vertex_types.h>
#include <Demo.h>
#include <Font.h>

void Font::Release()
{
  SAFE_DELETE_ARRAY(texCoords);
}

bool Font::Load(const char *fileName)
{
  // load ".font" file
  strcpy(name, fileName);
  char filePath[DEMO_MAX_FILEPATH];
  Demo::fileManager->GetFilePath(fileName, filePath);
  FILE *file;
  fopen_s(&file, filePath, "rb");
  if(!file)
    return false;

  // check idString
  char idString[10];
  memset(idString, 0, 10);
  fread(idString, sizeof(char), 9, file);
  if(strcmp(idString, "DEMO_FONT") != 0)
  {
    fclose(file);
    return false;
  }

  // check version
  unsigned int version;
  fread(&version, sizeof(unsigned int), 1, file);
  if(version != CURRENT_FONT_VERSION)
  {
    fclose(file);
    return false;
  }

  // load material
  char fontMaterialName[256];
  fread(fontMaterialName, sizeof(char), 256, file);
  material = Demo::resourceManager->LoadMaterial(fontMaterialName);
  if(!material)
  {
    fclose(file);
    return false;
  }

  // load font parameters 
  fread(&textureWidth, sizeof(unsigned int), 1, file);
  fread(&textureHeight, sizeof(unsigned int), 1, file);
  fread(&fontHeight, sizeof(unsigned int), 1, file);
  fread(&fontSpacing, sizeof(unsigned int), 1, file);

  // get number of texCoords
  fread(&numTexCoords, sizeof(unsigned int), 1, file);
  if(numTexCoords < 1)
  {
    fclose(file);
    return false;
  }

  // load texCoords
  texCoords = new float[numTexCoords];
  if(!texCoords)
  {
    fclose(file);
    return false;
  }
  fread(texCoords, sizeof(float), numTexCoords, file);

  fclose(file);

  // create vertex layout
  VertexElementDesc vertexElementDescs[3] = { POSITION_ATTRIB, R32G32B32_FLOAT_EF, 0,
                                              TEXCOORDS_ATTRIB, R32G32_FLOAT_EF, 12,
                                              COLOR_ATTRIB, R32G32B32_FLOAT_EF, 20 };
  vertexLayout = Demo::renderer->CreateVertexLayout(vertexElementDescs, 3);
  if(!vertexLayout)
  {
    SAFE_DELETE_ARRAY(texCoords);
    return false;
  }

  // create dynamic vertex buffer
  vertexBuffer = Demo::renderer->CreateVertexBuffer(sizeof(FontVertex), FONT_MAX_VERTEX_COUNT, true);
  if(!vertexBuffer)
  {
    SAFE_DELETE_ARRAY(texCoords);
    return false;
  }

  return true;
}

void Font::Print(const Vector2 &position, float scale, const Color &color, const char *string, ...)
{
  char str[FONT_MAX_TEXT_LENGTH];
  va_list va;
  if(!string)
    return;
  va_start(va, string);
  unsigned int length = _vscprintf(string, va)+1;
  if(length > FONT_MAX_TEXT_LENGTH) 
  { 
    va_end(va);
    return;
  }
  vsprintf_s(str, string, va);
  va_end(va);

  char *text = str;
  float positionX = position.x;
  float positionY = position.y;
  positionX -= (float)(fontSpacing/fontHeight)+(scale*0.5f);
  const float startX = positionX; 
  float aspectRatio = Demo::renderer->GetCamera(MAIN_CAMERA_ID)->GetAspectRatio();
  const float scaleX = scale;
  const float scaleY = scale*aspectRatio;
  const int maxCharIndex = numTexCoords/4;
  FontVertex vertices[2]; 
  while(*text)
  {
    char c = *text++; 
    if(c == '\n')
    {
      positionX = startX;
      positionY -= ((texCoords[3]-texCoords[1])*textureHeight/(float)fontHeight)*scaleY;
    }
    int charIndex = c-32;
    if((charIndex < 0) || (charIndex >= maxCharIndex))
      continue;
    float tx1 = texCoords[charIndex*4];
    float ty1 = texCoords[charIndex*4+3];
    float tx2 = texCoords[charIndex*4+2];  
    float ty2 = texCoords[charIndex*4+1];
    float width = ((tx2-tx1)*textureWidth/(float)fontHeight)*scaleX;
    float height = ((ty1-ty2)*textureHeight/(float)fontHeight)*scaleY;
    if(c != ' ')
    {  
      vertices[0].position = Vector3(positionX, positionY, 0.0f);
      vertices[0].texCoords = Vector2(tx1, ty1);
      vertices[0].color.Set(color.r, color.g, color.b);
      vertices[1].position = Vector3(positionX+width, positionY+height, 0.0f);
      vertices[1].texCoords = Vector2(tx2, ty2);
      vertices[1].color.Set(color.r, color.g, color.b);
      vertexBuffer->AddVertices(2, vertices);
    }

    positionX += width-(2.0f*fontSpacing*scaleX)/(float)fontHeight;
  }
}

void Font::AddSurfaces() 
{
  GpuCmd gpuCmd(DRAW_CM);	
  gpuCmd.order = GUI_CO;
  gpuCmd.draw.renderTarget = Demo::renderer->GetRenderTarget(BACK_BUFFER_RT_ID);
  gpuCmd.draw.vertexLayout = vertexLayout;
  gpuCmd.draw.vertexBuffer = vertexBuffer;
  gpuCmd.draw.primitiveType = LINES_PRIMITIVE;
  gpuCmd.draw.firstIndex = 0;
  gpuCmd.draw.numElements = vertexBuffer->GetVertexCount(); 
  gpuCmd.draw.textures[COLOR_TEX_ID] = material->colorTexture;
  gpuCmd.draw.samplers[COLOR_TEX_ID] = Demo::renderer->GetSampler(TRILINEAR_SAMPLER_ID);
  gpuCmd.draw.rasterizerState = material->rasterizerState;
  gpuCmd.draw.depthStencilState = material->depthStencilState;
  gpuCmd.draw.blendState = material->blendState;
  gpuCmd.draw.shader = material->shader;
  Demo::renderer->AddGpuCmd(gpuCmd);
}


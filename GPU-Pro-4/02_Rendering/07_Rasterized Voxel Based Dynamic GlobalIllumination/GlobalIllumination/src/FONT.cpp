#include <stdafx.h>
#include <vertex_types.h>
#include <DEMO.h>
#include <FONT.h>

void FONT::Release()
{
	SAFE_DELETE_ARRAY(texCoords);
}

bool FONT::Load(const char *fileName)
{
	// load ".font" file
	strcpy(name,fileName);
	char filePath[DEMO_MAX_FILEPATH];
	DEMO::fileManager->GetFilePath(fileName,filePath);
	FILE *file;
	fopen_s(&file,filePath,"rb");
	if(!file)
		return false;

	// check idString
	char idString[10];
	memset(idString,0,10);
	fread(idString,sizeof(char),9,file);
	if(strcmp(idString,"DEMO_FONT")!=0)
	{
		fclose(file);
		return false;
	}

	// check version
	int version;
	fread(&version,sizeof(int),1,file);
	if(version!=CURRENT_FONT_VERSION)
	{
		fclose(file);
		return false;
	}

	// load material
	char fontMaterialName[256];
	fread(fontMaterialName,sizeof(char),256,file);
	material = DEMO::resourceManager->LoadMaterial(fontMaterialName);
	if(!material)
	{
		fclose(file);
		return false;
	}

	// load font parameters 
	fread(&textureWidth,sizeof(int),1,file);
	fread(&textureHeight,sizeof(int),1,file);
	fread(&fontHeight,sizeof(int),1,file);
	fread(&fontSpacing,sizeof(int),1,file);

	// get number of texCoords
	fread(&numTexCoords,sizeof(int),1,file);
	if(numTexCoords<1)
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
	fread(texCoords,sizeof(float),numTexCoords,file);

	fclose(file);
	
	// render only into the accumulation render-target of GBuffer
	RT_CONFIG_DESC rtcDesc;
	rtcDesc.numColorBuffers = 1;
	rtConfig = DEMO::renderer->CreateRenderTargetConfig(rtcDesc);
	if(!rtConfig)
	{
		SAFE_DELETE_ARRAY(texCoords);
		return false;
	}

	// create dynamic vertex-buffer
	VERTEX_ELEMENT_DESC vertexLayout[3] = { POSITION_ELEMENT,R32G32B32_FLOAT_EF,0,
																				  TEXCOORDS_ELEMENT,R32G32_FLOAT_EF,3,
																					COLOR_ELEMENT,R32G32B32A32_FLOAT_EF,5 };
	vertexBuffer = DEMO::renderer->CreateVertexBuffer(vertexLayout,3,true,FONT_MAX_VERTEX_COUNT);
	if(!vertexBuffer)
	{
		SAFE_DELETE_ARRAY(texCoords);
		return false;
	}

	return true;
}

void FONT::Print(const VECTOR2D &position,float scale,const COLOR &color,const char *string,...)
{
	char str[FONT_MAX_TEXT_LENGTH];
	va_list va;
	if(!string)
		return;
	va_start(va,string);
	int length = _vscprintf(string,va)+1;
	if(length>FONT_MAX_TEXT_LENGTH) 
	{ 
		va_end(va);
		return;
	}
	vsprintf_s(str,string,va);
	va_end(va);

	char *text = str;
	float positionX = position.x;
	float positionY = position.y;
	positionX -= (fontSpacing/fontHeight)+(scale/2);
	float startX = positionX; 
	float aspectRatio = DEMO::renderer->GetCamera(MAIN_CAMERA_ID)->GetAspectRatio();
	FONT_VERTEX vertices[2]; 
	while(*text)
	{
		char c = *text++; 
		if(c=='\n')
		{
			positionX = startX;
			positionY -= ((texCoords[3]-texCoords[1])*textureHeight/(float)textureHeight)*scale;
		}
		int charIndex = c-32;
		if((charIndex<0)||(charIndex>=(numTexCoords/4)))
			continue;
		float tx1 = texCoords[charIndex*4];
		float ty1 = texCoords[charIndex*4+3];
		float tx2 = texCoords[charIndex*4+2];  
		float ty2 = texCoords[charIndex*4+1];
		float width = ((tx2-tx1)*textureWidth/(float)fontHeight)*scale;
		float height = ((ty1-ty2)*textureHeight/(float)fontHeight)*scale*aspectRatio;
		if(c!=' ')
		{  
			vertices[0].position = VECTOR3D(positionX,positionY,0.0f);
			vertices[0].texCoords = VECTOR2D(tx1,ty1);
			vertices[0].color = color;
			vertices[1].position = VECTOR3D(positionX+width,positionY+height,0.0f);
			vertices[1].texCoords = VECTOR2D(tx2,ty2);
			vertices[1].color = color;
			vertexBuffer->AddVertices(2,(float*)vertices);
		}

		positionX += width-(2*fontSpacing*scale)/(float)fontHeight;
	}
}

void FONT::AddSurfaces() 
{
	vertexBuffer->Update();

	SURFACE surface;
	surface.renderTarget = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID);
	surface.renderTargetConfig = rtConfig;
	surface.renderOrder = GUI_RO;
	surface.vertexBuffer = vertexBuffer;
	surface.primitiveType = LINES_PRIMITIVE;
	surface.firstIndex = 0;
	surface.numElements = vertexBuffer->GetVertexCount(); 
	surface.colorTexture = material->colorTexture;
	surface.rasterizerState = material->rasterizerState;
	surface.depthStencilState = material->depthStencilState;
	surface.blendState = material->blendState;
	surface.shader = material->shader;
	surface.renderMode = NON_INDEXED_RM;
	DEMO::renderer->AddSurface(surface);

	vertexBuffer->Clear();
}


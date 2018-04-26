/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

#include "DXUT.h"
#include "TextureContainer.h"

TextureContainer::TextureContainer(void)
{
}

TextureContainer::~TextureContainer(void)
{
}

HRESULT TextureContainer::addTexture(const TextureName &name, ID3D10Device* device)
{
	HRESULT hr = D3DX10CreateShaderResourceViewFromFile( device, name, NULL, NULL, 
         &textures[name], NULL );
	return hr;
}

void TextureContainer::releaseAll()
{
	for ( NameToTextureMap::iterator it = textures.begin(); it != textures.end(); ++it )
	{
		if ( it->second )
			it->second->Release();
	}
}

Texture *TextureContainer::getTexture( const TextureName &name )
{
	if ( textures.count( name ) == 0 )
		return NULL;
	return textures[name];
}

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

#pragma once

#pragma warning(disable:4995)		// disable warnings indicating deprecated codes in cstdio
#include <map>
#pragma warning(default:4995)

#include "UnicodeString.h"

struct ID3D10ShaderResourceView;
typedef UnicodeString TextureName;
typedef ID3D10ShaderResourceView Texture;
typedef std::map<TextureName, Texture*> NameToTextureMap;

struct ID3D10Device;

class TextureContainer
{
public:
	TextureContainer(void);
	~TextureContainer(void);

	HRESULT addTexture( const TextureName &name, ID3D10Device* device );
	void releaseAll();

	Texture *getTexture( const TextureName &name );

protected:
	NameToTextureMap textures;
};

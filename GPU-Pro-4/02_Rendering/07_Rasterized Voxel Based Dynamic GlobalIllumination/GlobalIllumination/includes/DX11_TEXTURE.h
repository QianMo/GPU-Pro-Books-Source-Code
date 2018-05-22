#ifndef DX11_TEXTURE_H
#define DX11_TEXTURE_H

#include <render_states.h>
#include <DX11_SAMPLER.h>

// DX11_TEXTURE
//   Manages a texture.
class DX11_TEXTURE 
{
public:
	friend class DX11_RENDER_TARGET; 

	DX11_TEXTURE()
	{
		sampler = NULL;
		texture = NULL;
		shaderResourceView = NULL;
		unorderedAccessView = NULL;
	}

	~DX11_TEXTURE()
	{
		Release();
	}

	void Release();	

	bool LoadFromFile(const char *fileName,DX11_SAMPLER *sampler=NULL); 

	// creates render-target texture
	bool CreateRenderable(int width,int height,int depth,texFormats format,DX11_SAMPLER *sampler=NULL,bool useUAV=false);	

	void Bind(textureBP bindingPoint,shaderTypes shaderType=VERTEX_SHADER) const;

	ID3D11UnorderedAccessView* GetUnorderdAccessView() const
	{
		return unorderedAccessView;
	}

	DX11_SAMPLER* GetSampler() const
	{
		return sampler;
	}

	const char* GetName() const
	{
		return name;
	}

private:	  
	char name[DEMO_MAX_FILENAME];
	DX11_SAMPLER *sampler;

	ID3D11Resource *texture;
	ID3D11ShaderResourceView *shaderResourceView;
	ID3D11UnorderedAccessView *unorderedAccessView;

};

#endif

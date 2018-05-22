#ifndef DX11_UNIFORM_BUFFER_H
#define DX11_UNIFORM_BUFFER_H

#include <LIST.h>  
#include <render_states.h>
#include <DX11_SHADER.h>

enum dataTypes
{
	INT_DT=0,
	FLOAT_DT,
	VEC2_DT,
	VEC3_DT,
	VEC4_DT,
	MAT4_DT
};

struct UNIFORM
{
	char name[64];
	dataTypes dataType;
	int count;
};

// UNIFORM_LIST
//   Convenient wrapper for a UNIFORM list.
class UNIFORM_LIST
{
public:
	UNIFORM_LIST()
	{
	}

	~UNIFORM_LIST()
	{
		uniforms.Erase();
	}

	// adds a new uniform to list
	void AddElement(const char* name,dataTypes dataType,int count=1)
	{
		UNIFORM uniform;
		strcpy(uniform.name,name);
		uniform.dataType = dataType;
		uniform.count = count;
		uniforms.AddElement(&uniform);
	}

	// gets a uniform from list by index 
	UNIFORM* GetElement(int index) const
	{
		if((index<0)||(index>=uniforms.GetSize()))
			return NULL;
		return &uniforms[index];
	}

	// gets number of uniforms
	int GetSize() const
	{
		return uniforms.GetSize(); 
	}

private:
	LIST<UNIFORM> uniforms;

};

// DX11_UNIFORM_BUFFER
//   Manages a uniform buffer (= constant buffer).
class DX11_UNIFORM_BUFFER
{
public:
	DX11_UNIFORM_BUFFER()
	{
		bindingPoint = 0;
		size = 0;
		uniformBuffer = NULL;	
	}

	~DX11_UNIFORM_BUFFER()
	{
		Release();
	}

	void Release();

	bool Create(uniformBufferBP bindingPoint,const UNIFORM_LIST &uniformList);

	// Please note: uniforms must be aligned according to the HLSL rules, in order to be able
	// to upload data in 1 block.
	bool Update(float *uniformBufferData);

	void Bind(shaderTypes shaderType=VERTEX_SHADER) const;

	int GetBindingPoint() const
	{
		return bindingPoint;
	}

private:
	int bindingPoint; // shader binding point
	int size; // size of uniform data
	ID3D11Buffer *uniformBuffer;  

};

#endif
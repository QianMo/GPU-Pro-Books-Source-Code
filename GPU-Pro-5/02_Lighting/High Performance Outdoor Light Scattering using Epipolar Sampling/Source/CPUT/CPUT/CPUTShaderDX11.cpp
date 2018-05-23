//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------

#include "CPUTShaderDX11.h"
#include "D3DCompiler.h"
#include "CPUTConfigBlock.h"
#include "CPUTMaterial.h"

//-----------------------------------------------------------------------------
bool CPUTShaderDX11::ShaderRequiresPerModelPayload( CPUTConfigBlock &properties )
{
    ID3D11ShaderReflection *pReflector = NULL;

    D3DReflect( mpBlob->GetBufferPointer(), mpBlob->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&pReflector);
    // Walk through the shader input bind descriptors.
    // If any of them begin with '@', then we need a unique material per model (i.e., we need to clone the material).
    int ii=0;
    D3D11_SHADER_INPUT_BIND_DESC desc;
	HRESULT hr = pReflector->GetResourceBindingDesc( ii++, &desc );
	while( SUCCEEDED(hr) )
	{
        cString tagName = s2ws(desc.Name);
        CPUTConfigEntry *pValue = properties.GetValueByName(tagName);
        if( !pValue->IsValid() )
        {
            // We didn't find our property in the file.  Is it in the global config block?
            pValue = CPUTMaterial::mGlobalProperties.GetValueByName(tagName);
        }
        cString boundName = pValue->ValueAsString();
        if( (boundName.length() > 0) && ((boundName[0] == '@') || (boundName[0] == '#')) )
        {
            return true;
        }
        hr = pReflector->GetResourceBindingDesc( ii++, &desc );
    }
    return false;
}

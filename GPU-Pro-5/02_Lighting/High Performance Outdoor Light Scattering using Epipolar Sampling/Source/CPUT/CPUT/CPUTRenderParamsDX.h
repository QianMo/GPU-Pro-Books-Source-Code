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
#ifndef __CPUTRENDERPARAMSDX_H__
#define __CPUTRENDERPARAMSDX_H__

#include "CPUT.h"
#include <d3d11.h>
#include <xnamath.h>
#include "CPUTRenderParams.h"

class CPUTRenderParametersDX : public CPUTRenderParameters
{
public:
    ID3D11DeviceContext *mpContext;

public:
    CPUTRenderParametersDX(): mpContext(NULL){}
    CPUTRenderParametersDX( ID3D11DeviceContext *pContext, bool drawModels=true, bool renderOnlyVisibleModels=true, bool showBoundingBoxes=false )
        : mpContext(pContext)
    {
        mShowBoundingBoxes       = showBoundingBoxes;
        mDrawModels              = drawModels;
        mRenderOnlyVisibleModels = renderOnlyVisibleModels;
    }
    ~CPUTRenderParametersDX(){}
};

#endif // #ifndef __CPUTRENDERPARAMSDX_H__
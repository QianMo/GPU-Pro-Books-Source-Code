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
#ifndef __CPUTRENDERPARAMS_H__
#define __CPUTRENDERPARAMS_H__

// TODO:  Change name to CPUTRenderContext?
class CPUTCamera;

class CPUTRenderParameters
{
public:
    bool         mShowBoundingBoxes;
    bool         mDrawModels;
    bool         mRenderOnlyVisibleModels;
    CPUTCamera  *mpCamera;

    CPUTRenderParameters() :
        mShowBoundingBoxes(false),
        mDrawModels(true),
        mRenderOnlyVisibleModels(true),
        mpCamera(0)
    {}
    ~CPUTRenderParameters(){}
private:
};

#endif // __CPUTRENDERPARAMS_H__
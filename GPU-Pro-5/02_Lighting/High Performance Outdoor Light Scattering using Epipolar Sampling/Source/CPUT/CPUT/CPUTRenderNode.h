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
#ifndef __CPUTRENDERNODE_H__
#define __CPUTRENDERNODE_H__

#include "CPUTRenderParams.h"
#include "CPUTRefCount.h"
#include "CPUTMath.h"
#include "CPUTConfigBlock.h"

// forward declarations
class CPUTCamera;

class CPUTRenderNode : public CPUTRefCount
{
protected:
    cString             mName;
    CPUTRenderNode     *mpParent;
    CPUTRenderNode     *mpChild;
    CPUTRenderNode     *mpSibling;
    bool                mWorldMatrixDirty;
    float4x4            mWorldMatrix; // transform of this object combined with it's parent(s) transform(s)
    float4x4            mParentMatrix;   // transform of this object relative to it's parent
    cString             mPrefix;
    ~CPUTRenderNode(); // Destructor is not public.  Must release instead of delete.

public:
    CPUTRenderNode();

    void Scale(float xx, float yy, float zz)
    {
        float4x4 scale(
              xx, 0.0f, 0.0f, 0.0f,
            0.0f,   yy, 0.0f, 0.0f,
            0.0f, 0.0f,   zz, 0.0f,
            0.0f, 0.0f, 0.0f,    1
        );
        mParentMatrix = mParentMatrix * scale;
        MarkDirty();
    }
    void Scale(float xx)
    {
        float4x4 scale(
           xx, 0.0f, 0.0f, 0.0f,
         0.0f,   xx, 0.0f, 0.0f,
         0.0f, 0.0f,   xx, 0.0f,
         0.0f, 0.0f, 0.0f,    1
        );
        mParentMatrix = mParentMatrix * scale;
        MarkDirty();
    }
    void SetPosition(float x, float y, float z)
    {
        mParentMatrix.r3.x = x;
        mParentMatrix.r3.y = y;
        mParentMatrix.r3.z = z;
        MarkDirty();
    }
    void SetPosition(float3 &position)
    {
        mParentMatrix.r3.x = position.x;
        mParentMatrix.r3.y = position.y;
        mParentMatrix.r3.z = position.z;
        MarkDirty();
    }
    void GetPosition(float *pX, float *pY, float *pZ)
    {
        *pX = mParentMatrix.r3.x;
        *pY = mParentMatrix.r3.y;
        *pZ = mParentMatrix.r3.z;
    }
    void GetPosition(float3 *pPosition)
    {
        pPosition->x = mParentMatrix.r3.x;
        pPosition->y = mParentMatrix.r3.y;
        pPosition->z = mParentMatrix.r3.z;
    }
    float3 GetPosition()
    {
        float3 ret = float3( mParentMatrix.r3.x, mParentMatrix.r3.y, mParentMatrix.r3.z);
        return ret;
    }
    float3 GetLook()
    {
        return mParentMatrix.getZAxis();
    }
    float3 GetUp()
    {
        return mParentMatrix.getYAxis();
    }
    float3 GetLook( float *pX, float *pY, float *pZ )
    {
        float3 look = mParentMatrix.getZAxis();
        *pX = look.x;
        *pY = look.y;
        *pZ = look.z;
    }

    virtual int      ReleaseRecursive();
    void             SetName(const cString &name) { mName = name;};
    void             SetParent(CPUTRenderNode *pParent);
    CPUTRenderNode  *GetParent()       { return mpParent; }
    CPUTRenderNode  *GetChild()        { return mpChild; }
    CPUTRenderNode  *GetSibling()      { return mpSibling; }
    cString         &GetName()         { return mName;};
    cString         &GetPrefix()       { return mPrefix; }
    void             SetPrefix( cString &prefix ) { mPrefix = prefix; }
    virtual bool     IsModel()         { return false; }
    float4x4        *GetParentMatrix() { return &mParentMatrix; }
    float4x4        *GetWorldMatrix();
    void             MarkDirty();
    void             AddChild(CPUTRenderNode *pNode);
    void             AddSibling(CPUTRenderNode *pNode);
    virtual void     Update( float deltaSeconds = 0.0f ){}
    virtual void     UpdateRecursive( float deltaSeconds );
    virtual void     Render(CPUTRenderParameters &renderParams){}
    virtual void     RenderShadow(CPUTRenderParameters &renderParams){}
    virtual void     RenderRecursive(CPUTRenderParameters &renderParams);
    virtual void     RenderShadowRecursive(CPUTRenderParameters &renderParams);
    void             SetParentMatrix(const float4x4 &parentMatrix)
    {
        mParentMatrix = parentMatrix;
        MarkDirty();
    }
    void LoadParentMatrixFromParameterBlock( CPUTConfigBlock *pBlock )
    {
        // get and set the transform
        float pMatrix[4][4];
        CPUTConfigEntry *pColumn = pBlock->GetValueByName(_L("matrixColumn0"));
        CPUTConfigEntry *pRow    = pBlock->GetValueByName(_L("matrixRow0"));
        float4x4 parentMatrix;
        if( pColumn->IsValid() )
        {
            pBlock->GetValueByName(_L("matrixColumn0"))->ValueAsFloatArray(&pMatrix[0][0], 4);
            pBlock->GetValueByName(_L("matrixColumn1"))->ValueAsFloatArray(&pMatrix[1][0], 4);
            pBlock->GetValueByName(_L("matrixColumn2"))->ValueAsFloatArray(&pMatrix[2][0], 4);
            pBlock->GetValueByName(_L("matrixColumn3"))->ValueAsFloatArray(&pMatrix[3][0], 4);
            parentMatrix = float4x4((float*)&pMatrix[0][0]);
            parentMatrix.transpose(); // Matrices are specified column-major, but consumed row-major.
        } else if( pRow->IsValid() )
        {
            pBlock->GetValueByName(_L("matrixRow0"))->ValueAsFloatArray(&pMatrix[0][0], 4);
            pBlock->GetValueByName(_L("matrixRow1"))->ValueAsFloatArray(&pMatrix[1][0], 4);
            pBlock->GetValueByName(_L("matrixRow2"))->ValueAsFloatArray(&pMatrix[2][0], 4);
            pBlock->GetValueByName(_L("matrixRow3"))->ValueAsFloatArray(&pMatrix[3][0], 4);
            parentMatrix = float4x4((float*)&pMatrix[0][0]);
        } else
        {
            float identity[16] = { 1.f, 0.f, 0.f, 0.f,   0.f, 1.f, 0.f, 0.f,   0.f, 0.f, 1.f, 0.f,    0.f, 0.f, 0.f, 1.f };
            parentMatrix = float4x4(identity);
        }
        SetParentMatrix(parentMatrix);   // set the relative transform, marking child transform's dirty
    }
    virtual void GetBoundingBoxRecursive( float3 *pCenter, float3 *pHalf)
    {
        if(mpChild)   { mpChild->GetBoundingBoxRecursive(   pCenter, pHalf ); }
        if(mpSibling) { mpSibling->GetBoundingBoxRecursive( pCenter, pHalf ); }
    }
};

#endif // #ifndef __CPUTRENDERNODE_H__

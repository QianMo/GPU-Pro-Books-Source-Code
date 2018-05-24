//--------------------------------------------------------------------------------------
// File: SceneResources.cpp
//
// This is where the resources are allocated and freed
// This sample is based off Microsoft DirectX SDK sample CascadedShadowMap11
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------


#include "dxut.h"

#include <DirectXColors.h>
#include "SceneManager.h"
#include "DXUTcamera.h"
#include "SDKMesh.h"
#include "DirectXCollision.h"
#include "SDKmisc.h"

using namespace DirectX;

static const uint8_t     g_ZeroMemory[512]  = { 0 };

//--------------------------------------------------------------------------------------------------
// Pre-built table for an ellipsoid shell
static __declspec(align(16)) float s_EllipsoidVertTable[][4] = 
{
  { -0.00000, -1.03528, -0.00000, 0.0 },  { 0.53590, -0.89658, 0.00000, 0.0 },  { -0.00000, -1.03528, -0.00000, 0.0 },
  { -0.00000, -1.03528, -0.00000, 0.0 },  { 0.53590, -0.89658, 0.00000, 0.0 },  { 0.46410, -0.89658, 0.26795, 0.0 },
  { -0.00000, -1.03528, -0.00000, 0.0 },  { 0.46410, -0.89658, 0.26795, 0.0 },  { -0.00000, -1.03528, -0.00000, 0.0 },
  { -0.00000, -1.03528, -0.00000, 0.0 },  { 0.46410, -0.89658, 0.26795, 0.0 },  { 0.26795, -0.89658, 0.46410, 0.0 },
  { -0.00000, -1.03528, -0.00000, 0.0 },  { 0.26795, -0.89658, 0.46410, 0.0 },  { 0.00000, -1.03528, -0.00000, 0.0 },
  { 0.00000, -1.03528, -0.00000, 0.0 },  { 0.26795, -0.89658, 0.46410, 0.0 },  { -0.00000, -0.89658, 0.53590, 0.0 },
  { 0.00000, -1.03528, -0.00000, 0.0 },  { -0.00000, -0.89658, 0.53590, 0.0 },  { 0.00000, -1.03528, -0.00000, 0.0 },
  { 0.00000, -1.03528, -0.00000, 0.0 },  { -0.00000, -0.89658, 0.53590, 0.0 },  { -0.26795, -0.89658, 0.46410, 0.0 },
  { 0.00000, -1.03528, -0.00000, 0.0 },  { -0.26795, -0.89658, 0.46410, 0.0 },  { 0.00000, -1.03528, -0.00000, 0.0 },
  { 0.00000, -1.03528, -0.00000, 0.0 },  { -0.26795, -0.89658, 0.46410, 0.0 },  { -0.46410, -0.89658, 0.26795, 0.0 },
  { 0.00000, -1.03528, -0.00000, 0.0 },  { -0.46410, -0.89658, 0.26795, 0.0 },  { 0.00000, -1.03528, 0.00000, 0.0 },
  { 0.00000, -1.03528, 0.00000, 0.0 },  { -0.46410, -0.89658, 0.26795, 0.0 },  { -0.53590, -0.89658, -0.00000, 0.0 },
  { 0.00000, -1.03528, 0.00000, 0.0 },  { -0.53590, -0.89658, -0.00000, 0.0 },  { 0.00000, -1.03528, 0.00000, 0.0 },
  { 0.00000, -1.03528, 0.00000, 0.0 },  { -0.53590, -0.89658, -0.00000, 0.0 },  { -0.46410, -0.89658, -0.26795, 0.0 },
  { 0.00000, -1.03528, 0.00000, 0.0 },  { -0.46410, -0.89658, -0.26795, 0.0 },  { 0.00000, -1.03528, 0.00000, 0.0 },
  { 0.00000, -1.03528, 0.00000, 0.0 },  { -0.46410, -0.89658, -0.26795, 0.0 },  { -0.26795, -0.89658, -0.46410, 0.0 },
  { 0.00000, -1.03528, 0.00000, 0.0 },  { -0.26795, -0.89658, -0.46410, 0.0 },  { -0.00000, -1.03528, 0.00000, 0.0 },
  { -0.00000, -1.03528, 0.00000, 0.0 },  { -0.26795, -0.89658, -0.46410, 0.0 },  { 0.00000, -0.89658, -0.53590, 0.0 },
  { -0.00000, -1.03528, 0.00000, 0.0 },  { 0.00000, -0.89658, -0.53590, 0.0 },  { -0.00000, -1.03528, 0.00000, 0.0 },
  { -0.00000, -1.03528, 0.00000, 0.0 },  { 0.00000, -0.89658, -0.53590, 0.0 },  { 0.26795, -0.89658, -0.46410, 0.0 },
  { -0.00000, -1.03528, 0.00000, 0.0 },  { 0.26795, -0.89658, -0.46410, 0.0 },  { -0.00000, -1.03528, 0.00000, 0.0 },
  { -0.00000, -1.03528, 0.00000, 0.0 },  { 0.26795, -0.89658, -0.46410, 0.0 },  { 0.46410, -0.89658, -0.26795, 0.0 },
  { -0.00000, -1.03528, 0.00000, 0.0 },  { 0.46410, -0.89658, -0.26795, 0.0 },  { -0.00000, -1.03528, -0.00000, 0.0 },
  { -0.00000, -1.03528, -0.00000, 0.0 },  { 0.46410, -0.89658, -0.26795, 0.0 },  { 0.53590, -0.89658, 0.00000, 0.0 },
  { 0.53590, -0.89658, 0.00000, 0.0 },  { 0.92820, -0.51764, 0.00000, 0.0 },  { 0.46410, -0.89658, 0.26795, 0.0 },
  { 0.46410, -0.89658, 0.26795, 0.0 },  { 0.92820, -0.51764, 0.00000, 0.0 },  { 0.80385, -0.51764, 0.46410, 0.0 },
  { 0.46410, -0.89658, 0.26795, 0.0 },  { 0.80385, -0.51764, 0.46410, 0.0 },  { 0.26795, -0.89658, 0.46410, 0.0 },
  { 0.26795, -0.89658, 0.46410, 0.0 },  { 0.80385, -0.51764, 0.46410, 0.0 },  { 0.46410, -0.51764, 0.80385, 0.0 },
  { 0.26795, -0.89658, 0.46410, 0.0 },  { 0.46410, -0.51764, 0.80385, 0.0 },  { -0.00000, -0.89658, 0.53590, 0.0 },
  { -0.00000, -0.89658, 0.53590, 0.0 },  { 0.46410, -0.51764, 0.80385, 0.0 },  { -0.00000, -0.51764, 0.92820, 0.0 },
  { -0.00000, -0.89658, 0.53590, 0.0 },  { -0.00000, -0.51764, 0.92820, 0.0 },  { -0.26795, -0.89658, 0.46410, 0.0 },
  { -0.26795, -0.89658, 0.46410, 0.0 },  { -0.00000, -0.51764, 0.92820, 0.0 },  { -0.46410, -0.51764, 0.80385, 0.0 },
  { -0.26795, -0.89658, 0.46410, 0.0 },  { -0.46410, -0.51764, 0.80385, 0.0 },  { -0.46410, -0.89658, 0.26795, 0.0 },
  { -0.46410, -0.89658, 0.26795, 0.0 },  { -0.46410, -0.51764, 0.80385, 0.0 },  { -0.80385, -0.51764, 0.46410, 0.0 },
  { -0.46410, -0.89658, 0.26795, 0.0 },  { -0.80385, -0.51764, 0.46410, 0.0 },  { -0.53590, -0.89658, -0.00000, 0.0 },
  { -0.53590, -0.89658, -0.00000, 0.0 },  { -0.80385, -0.51764, 0.46410, 0.0 },  { -0.92820, -0.51764, -0.00000, 0.0 },
  { -0.53590, -0.89658, -0.00000, 0.0 },  { -0.92820, -0.51764, -0.00000, 0.0 },  { -0.46410, -0.89658, -0.26795, 0.0 },
  { -0.46410, -0.89658, -0.26795, 0.0 },  { -0.92820, -0.51764, -0.00000, 0.0 },  { -0.80385, -0.51764, -0.46410, 0.0 },
  { -0.46410, -0.89658, -0.26795, 0.0 },  { -0.80385, -0.51764, -0.46410, 0.0 },  { -0.26795, -0.89658, -0.46410, 0.0 },
  { -0.26795, -0.89658, -0.46410, 0.0 },  { -0.80385, -0.51764, -0.46410, 0.0 },  { -0.46410, -0.51764, -0.80385, 0.0 },
  { -0.26795, -0.89658, -0.46410, 0.0 },  { -0.46410, -0.51764, -0.80385, 0.0 },  { 0.00000, -0.89658, -0.53590, 0.0 },
  { 0.00000, -0.89658, -0.53590, 0.0 },  { -0.46410, -0.51764, -0.80385, 0.0 },  { 0.00000, -0.51764, -0.92820, 0.0 },
  { 0.00000, -0.89658, -0.53590, 0.0 },  { 0.00000, -0.51764, -0.92820, 0.0 },  { 0.26795, -0.89658, -0.46410, 0.0 },
  { 0.26795, -0.89658, -0.46410, 0.0 },  { 0.00000, -0.51764, -0.92820, 0.0 },  { 0.46410, -0.51764, -0.80385, 0.0 },
  { 0.26795, -0.89658, -0.46410, 0.0 },  { 0.46410, -0.51764, -0.80385, 0.0 },  { 0.46410, -0.89658, -0.26795, 0.0 },
  { 0.46410, -0.89658, -0.26795, 0.0 },  { 0.46410, -0.51764, -0.80385, 0.0 },  { 0.80385, -0.51764, -0.46410, 0.0 },
  { 0.46410, -0.89658, -0.26795, 0.0 },  { 0.80385, -0.51764, -0.46410, 0.0 },  { 0.53590, -0.89658, 0.00000, 0.0 },
  { 0.53590, -0.89658, 0.00000, 0.0 },  { 0.80385, -0.51764, -0.46410, 0.0 },  { 0.92820, -0.51764, 0.00000, 0.0 },
  { 0.92820, -0.51764, 0.00000, 0.0 },  { 1.07180, 0.00000, 0.00000, 0.0 },  { 0.80385, -0.51764, 0.46410, 0.0 },
  { 0.80385, -0.51764, 0.46410, 0.0 },  { 1.07180, 0.00000, 0.00000, 0.0 },  { 0.92820, 0.00000, 0.53590, 0.0 },
  { 0.80385, -0.51764, 0.46410, 0.0 },  { 0.92820, 0.00000, 0.53590, 0.0 },  { 0.46410, -0.51764, 0.80385, 0.0 },
  { 0.46410, -0.51764, 0.80385, 0.0 },  { 0.92820, 0.00000, 0.53590, 0.0 },  { 0.53590, 0.00000, 0.92820, 0.0 },
  { 0.46410, -0.51764, 0.80385, 0.0 },  { 0.53590, 0.00000, 0.92820, 0.0 },  { -0.00000, -0.51764, 0.92820, 0.0 },
  { -0.00000, -0.51764, 0.92820, 0.0 },  { 0.53590, 0.00000, 0.92820, 0.0 },  { -0.00000, 0.00000, 1.07180, 0.0 },
  { -0.00000, -0.51764, 0.92820, 0.0 },  { -0.00000, 0.00000, 1.07180, 0.0 },  { -0.46410, -0.51764, 0.80385, 0.0 },
  { -0.46410, -0.51764, 0.80385, 0.0 },  { -0.00000, 0.00000, 1.07180, 0.0 },  { -0.53590, 0.00000, 0.92820, 0.0 },
  { -0.46410, -0.51764, 0.80385, 0.0 },  { -0.53590, 0.00000, 0.92820, 0.0 },  { -0.80385, -0.51764, 0.46410, 0.0 },
  { -0.80385, -0.51764, 0.46410, 0.0 },  { -0.53590, 0.00000, 0.92820, 0.0 },  { -0.92820, 0.00000, 0.53590, 0.0 },
  { -0.80385, -0.51764, 0.46410, 0.0 },  { -0.92820, 0.00000, 0.53590, 0.0 },  { -0.92820, -0.51764, -0.00000, 0.0 },
  { -0.92820, -0.51764, -0.00000, 0.0 },  { -0.92820, 0.00000, 0.53590, 0.0 },  { -1.07180, 0.00000, -0.00000, 0.0 },
  { -0.92820, -0.51764, -0.00000, 0.0 },  { -1.07180, 0.00000, -0.00000, 0.0 },  { -0.80385, -0.51764, -0.46410, 0.0 },
  { -0.80385, -0.51764, -0.46410, 0.0 },  { -1.07180, 0.00000, -0.00000, 0.0 },  { -0.92820, 0.00000, -0.53590, 0.0 },
  { -0.80385, -0.51764, -0.46410, 0.0 },  { -0.92820, 0.00000, -0.53590, 0.0 },  { -0.46410, -0.51764, -0.80385, 0.0 },
  { -0.46410, -0.51764, -0.80385, 0.0 },  { -0.92820, 0.00000, -0.53590, 0.0 },  { -0.53590, 0.00000, -0.92820, 0.0 },
  { -0.46410, -0.51764, -0.80385, 0.0 },  { -0.53590, 0.00000, -0.92820, 0.0 },  { 0.00000, -0.51764, -0.92820, 0.0 },
  { 0.00000, -0.51764, -0.92820, 0.0 },  { -0.53590, 0.00000, -0.92820, 0.0 },  { 0.00000, 0.00000, -1.07180, 0.0 },
  { 0.00000, -0.51764, -0.92820, 0.0 },  { 0.00000, 0.00000, -1.07180, 0.0 },  { 0.46410, -0.51764, -0.80385, 0.0 },
  { 0.46410, -0.51764, -0.80385, 0.0 },  { 0.00000, 0.00000, -1.07180, 0.0 },  { 0.53590, 0.00000, -0.92820, 0.0 },
  { 0.46410, -0.51764, -0.80385, 0.0 },  { 0.53590, 0.00000, -0.92820, 0.0 },  { 0.80385, -0.51764, -0.46410, 0.0 },
  { 0.80385, -0.51764, -0.46410, 0.0 },  { 0.53590, 0.00000, -0.92820, 0.0 },  { 0.92820, 0.00000, -0.53590, 0.0 },
  { 0.80385, -0.51764, -0.46410, 0.0 },  { 0.92820, 0.00000, -0.53590, 0.0 },  { 0.92820, -0.51764, 0.00000, 0.0 },
  { 0.92820, -0.51764, 0.00000, 0.0 },  { 0.92820, 0.00000, -0.53590, 0.0 },  { 1.07180, 0.00000, 0.00000, 0.0 },
  { 1.07180, 0.00000, 0.00000, 0.0 },  { 0.92820, 0.51764, 0.00000, 0.0 },  { 0.92820, 0.00000, 0.53590, 0.0 },
  { 0.92820, 0.00000, 0.53590, 0.0 },  { 0.92820, 0.51764, 0.00000, 0.0 },  { 0.80385, 0.51764, 0.46410, 0.0 },
  { 0.92820, 0.00000, 0.53590, 0.0 },  { 0.80385, 0.51764, 0.46410, 0.0 },  { 0.53590, 0.00000, 0.92820, 0.0 },
  { 0.53590, 0.00000, 0.92820, 0.0 },  { 0.80385, 0.51764, 0.46410, 0.0 },  { 0.46410, 0.51764, 0.80385, 0.0 },
  { 0.53590, 0.00000, 0.92820, 0.0 },  { 0.46410, 0.51764, 0.80385, 0.0 },  { -0.00000, 0.00000, 1.07180, 0.0 },
  { -0.00000, 0.00000, 1.07180, 0.0 },  { 0.46410, 0.51764, 0.80385, 0.0 },  { -0.00000, 0.51764, 0.92820, 0.0 },
  { -0.00000, 0.00000, 1.07180, 0.0 },  { -0.00000, 0.51764, 0.92820, 0.0 },  { -0.53590, 0.00000, 0.92820, 0.0 },
  { -0.53590, 0.00000, 0.92820, 0.0 },  { -0.00000, 0.51764, 0.92820, 0.0 },  { -0.46410, 0.51764, 0.80385, 0.0 },
  { -0.53590, 0.00000, 0.92820, 0.0 },  { -0.46410, 0.51764, 0.80385, 0.0 },  { -0.92820, 0.00000, 0.53590, 0.0 },
  { -0.92820, 0.00000, 0.53590, 0.0 },  { -0.46410, 0.51764, 0.80385, 0.0 },  { -0.80385, 0.51764, 0.46410, 0.0 },
  { -0.92820, 0.00000, 0.53590, 0.0 },  { -0.80385, 0.51764, 0.46410, 0.0 },  { -1.07180, 0.00000, -0.00000, 0.0 },
  { -1.07180, 0.00000, -0.00000, 0.0 },  { -0.80385, 0.51764, 0.46410, 0.0 },  { -0.92820, 0.51764, -0.00000, 0.0 },
  { -1.07180, 0.00000, -0.00000, 0.0 },  { -0.92820, 0.51764, -0.00000, 0.0 },  { -0.92820, 0.00000, -0.53590, 0.0 },
  { -0.92820, 0.00000, -0.53590, 0.0 },  { -0.92820, 0.51764, -0.00000, 0.0 },  { -0.80385, 0.51764, -0.46410, 0.0 },
  { -0.92820, 0.00000, -0.53590, 0.0 },  { -0.80385, 0.51764, -0.46410, 0.0 },  { -0.53590, 0.00000, -0.92820, 0.0 },
  { -0.53590, 0.00000, -0.92820, 0.0 },  { -0.80385, 0.51764, -0.46410, 0.0 },  { -0.46410, 0.51764, -0.80385, 0.0 },
  { -0.53590, 0.00000, -0.92820, 0.0 },  { -0.46410, 0.51764, -0.80385, 0.0 },  { 0.00000, 0.00000, -1.07180, 0.0 },
  { 0.00000, 0.00000, -1.07180, 0.0 },  { -0.46410, 0.51764, -0.80385, 0.0 },  { 0.00000, 0.51764, -0.92820, 0.0 },
  { 0.00000, 0.00000, -1.07180, 0.0 },  { 0.00000, 0.51764, -0.92820, 0.0 },  { 0.53590, 0.00000, -0.92820, 0.0 },
  { 0.53590, 0.00000, -0.92820, 0.0 },  { 0.00000, 0.51764, -0.92820, 0.0 },  { 0.46410, 0.51764, -0.80385, 0.0 },
  { 0.53590, 0.00000, -0.92820, 0.0 },  { 0.46410, 0.51764, -0.80385, 0.0 },  { 0.92820, 0.00000, -0.53590, 0.0 },
  { 0.92820, 0.00000, -0.53590, 0.0 },  { 0.46410, 0.51764, -0.80385, 0.0 },  { 0.80385, 0.51764, -0.46410, 0.0 },
  { 0.92820, 0.00000, -0.53590, 0.0 },  { 0.80385, 0.51764, -0.46410, 0.0 },  { 1.07180, 0.00000, 0.00000, 0.0 },
  { 1.07180, 0.00000, 0.00000, 0.0 },  { 0.80385, 0.51764, -0.46410, 0.0 },  { 0.92820, 0.51764, 0.00000, 0.0 },
  { 0.92820, 0.51764, 0.00000, 0.0 },  { 0.53590, 0.89658, 0.00000, 0.0 },  { 0.80385, 0.51764, 0.46410, 0.0 },
  { 0.80385, 0.51764, 0.46410, 0.0 },  { 0.53590, 0.89658, 0.00000, 0.0 },  { 0.46410, 0.89658, 0.26795, 0.0 },
  { 0.80385, 0.51764, 0.46410, 0.0 },  { 0.46410, 0.89658, 0.26795, 0.0 },  { 0.46410, 0.51764, 0.80385, 0.0 },
  { 0.46410, 0.51764, 0.80385, 0.0 },  { 0.46410, 0.89658, 0.26795, 0.0 },  { 0.26795, 0.89658, 0.46410, 0.0 },
  { 0.46410, 0.51764, 0.80385, 0.0 },  { 0.26795, 0.89658, 0.46410, 0.0 },  { -0.00000, 0.51764, 0.92820, 0.0 },
  { -0.00000, 0.51764, 0.92820, 0.0 },  { 0.26795, 0.89658, 0.46410, 0.0 },  { -0.00000, 0.89658, 0.53590, 0.0 },
  { -0.00000, 0.51764, 0.92820, 0.0 },  { -0.00000, 0.89658, 0.53590, 0.0 },  { -0.46410, 0.51764, 0.80385, 0.0 },
  { -0.46410, 0.51764, 0.80385, 0.0 },  { -0.00000, 0.89658, 0.53590, 0.0 },  { -0.26795, 0.89658, 0.46410, 0.0 },
  { -0.46410, 0.51764, 0.80385, 0.0 },  { -0.26795, 0.89658, 0.46410, 0.0 },  { -0.80385, 0.51764, 0.46410, 0.0 },
  { -0.80385, 0.51764, 0.46410, 0.0 },  { -0.26795, 0.89658, 0.46410, 0.0 },  { -0.46410, 0.89658, 0.26795, 0.0 },
  { -0.80385, 0.51764, 0.46410, 0.0 },  { -0.46410, 0.89658, 0.26795, 0.0 },  { -0.92820, 0.51764, -0.00000, 0.0 },
  { -0.92820, 0.51764, -0.00000, 0.0 },  { -0.46410, 0.89658, 0.26795, 0.0 },  { -0.53590, 0.89658, -0.00000, 0.0 },
  { -0.92820, 0.51764, -0.00000, 0.0 },  { -0.53590, 0.89658, -0.00000, 0.0 },  { -0.80385, 0.51764, -0.46410, 0.0 },
  { -0.80385, 0.51764, -0.46410, 0.0 },  { -0.53590, 0.89658, -0.00000, 0.0 },  { -0.46410, 0.89658, -0.26795, 0.0 },
  { -0.80385, 0.51764, -0.46410, 0.0 },  { -0.46410, 0.89658, -0.26795, 0.0 },  { -0.46410, 0.51764, -0.80385, 0.0 },
  { -0.46410, 0.51764, -0.80385, 0.0 },  { -0.46410, 0.89658, -0.26795, 0.0 },  { -0.26795, 0.89658, -0.46410, 0.0 },
  { -0.46410, 0.51764, -0.80385, 0.0 },  { -0.26795, 0.89658, -0.46410, 0.0 },  { 0.00000, 0.51764, -0.92820, 0.0 },
  { 0.00000, 0.51764, -0.92820, 0.0 },  { -0.26795, 0.89658, -0.46410, 0.0 },  { 0.00000, 0.89658, -0.53590, 0.0 },
  { 0.00000, 0.51764, -0.92820, 0.0 },  { 0.00000, 0.89658, -0.53590, 0.0 },  { 0.46410, 0.51764, -0.80385, 0.0 },
  { 0.46410, 0.51764, -0.80385, 0.0 },  { 0.00000, 0.89658, -0.53590, 0.0 },  { 0.26795, 0.89658, -0.46410, 0.0 },
  { 0.46410, 0.51764, -0.80385, 0.0 },  { 0.26795, 0.89658, -0.46410, 0.0 },  { 0.80385, 0.51764, -0.46410, 0.0 },
  { 0.80385, 0.51764, -0.46410, 0.0 },  { 0.26795, 0.89658, -0.46410, 0.0 },  { 0.46410, 0.89658, -0.26795, 0.0 },
  { 0.80385, 0.51764, -0.46410, 0.0 },  { 0.46410, 0.89658, -0.26795, 0.0 },  { 0.92820, 0.51764, 0.00000, 0.0 },
  { 0.92820, 0.51764, 0.00000, 0.0 },  { 0.46410, 0.89658, -0.26795, 0.0 },  { 0.53590, 0.89658, 0.00000, 0.0 },
  { 0.53590, 0.89658, 0.00000, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },  { 0.46410, 0.89658, 0.26795, 0.0 },
  { 0.46410, 0.89658, 0.26795, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },
  { 0.46410, 0.89658, 0.26795, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },  { 0.26795, 0.89658, 0.46410, 0.0 },
  { 0.26795, 0.89658, 0.46410, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },
  { 0.26795, 0.89658, 0.46410, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },  { -0.00000, 0.89658, 0.53590, 0.0 },
  { -0.00000, 0.89658, 0.53590, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },
  { -0.00000, 0.89658, 0.53590, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },  { -0.26795, 0.89658, 0.46410, 0.0 },
  { -0.26795, 0.89658, 0.46410, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },
  { -0.26795, 0.89658, 0.46410, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },  { -0.46410, 0.89658, 0.26795, 0.0 },
  { -0.46410, 0.89658, 0.26795, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },
  { -0.46410, 0.89658, 0.26795, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },  { -0.53590, 0.89658, -0.00000, 0.0 },
  { -0.53590, 0.89658, -0.00000, 0.0 },  { 0.00000, 1.03528, -0.00000, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },
  { -0.53590, 0.89658, -0.00000, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },  { -0.46410, 0.89658, -0.26795, 0.0 },
  { -0.46410, 0.89658, -0.26795, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },
  { -0.46410, 0.89658, -0.26795, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },  { -0.26795, 0.89658, -0.46410, 0.0 },
  { -0.26795, 0.89658, -0.46410, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },
  { -0.26795, 0.89658, -0.46410, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },  { 0.00000, 0.89658, -0.53590, 0.0 },
  { 0.00000, 0.89658, -0.53590, 0.0 },  { 0.00000, 1.03528, 0.00000, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },
  { 0.00000, 0.89658, -0.53590, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },  { 0.26795, 0.89658, -0.46410, 0.0 },
  { 0.26795, 0.89658, -0.46410, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },
  { 0.26795, 0.89658, -0.46410, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },  { 0.46410, 0.89658, -0.26795, 0.0 },
  { 0.46410, 0.89658, -0.26795, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },
  { 0.46410, 0.89658, -0.26795, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },  { 0.53590, 0.89658, 0.00000, 0.0 },
  { 0.53590, 0.89658, 0.00000, 0.0 },  { -0.00000, 1.03528, 0.00000, 0.0 },  { -0.00000, 1.03528, -0.00000, 0.0 },
};

//--------------------------------------------------------------------------------------
HRESULT CreateVertexShader(VertexShader* shader, ID3D11Device* pd3dDevice, LPCWSTR filename, LPCSTR main)
{
  HRESULT hr = S_OK;

  if ( !shader->m_ShaderBlob ) 
  {
    V_RETURN( DXUTCompileFromFile(filename, nullptr, main, "vs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &shader->m_ShaderBlob ) );
  }

  V_RETURN( pd3dDevice->CreateVertexShader(  shader->m_ShaderBlob->GetBufferPointer(), shader->m_ShaderBlob->GetBufferSize(),  nullptr, &shader->m_Shader ) );

  return hr;
}

//--------------------------------------------------------------------------------------
HRESULT CreatePixelShader(PixelShader* shader, ID3D11Device* pd3dDevice, LPCWSTR filename, LPCSTR main)
{
  HRESULT hr = S_OK;

  if ( !shader->m_ShaderBlob ) 
  {
    V_RETURN( DXUTCompileFromFile(filename, nullptr, main, "ps_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &shader->m_ShaderBlob ) );
  }

  V_RETURN( pd3dDevice->CreatePixelShader(  shader->m_ShaderBlob->GetBufferPointer(), shader->m_ShaderBlob->GetBufferSize(),  nullptr, &shader->m_Shader ) );

  return hr;
}


//--------------------------------------------------------------------------------------
SceneManager::SceneManager () 
                          : m_pVertexLayoutLight( nullptr ),
                            m_pVertexLayoutMesh( nullptr ),
                            m_pSamLinear( nullptr ),
                            m_pSamPoint( nullptr ),
                            m_pSamShadowPCF( nullptr ),   
                            m_fBlurBetweenCascadesAmount( 0.005f ),
                            m_RenderOneTileVP( m_RenderVP[0] ), 
                            m_iPCFBlurSize( 3 ),
                            m_fPCFOffset( 0.002f ),
                            m_DebugRendering(DEBUG_RENDERING_NONE),
                            m_DynamicLights(true)
{    
};

//--------------------------------------------------------------------------------------
// Create the resources, compile shaders, etc. 
//--------------------------------------------------------------------------------------
HRESULT SceneManager::Init ( ID3D11Device*           pd3dDevice,
                             ID3D11DeviceContext*    pd3dDeviceContext,
                             
                             CDXUTSDKMesh*           pMesh,

                             uint32_t                width,
                             uint32_t                height
                            )  
{
    HRESULT hr = S_OK; 
          
    m_ScratchSize             = 4 * 1024 * 1024; 
    m_ScratchBase             = malloc(m_ScratchSize);
    m_ScratchOffset           = 0;

    m_pd3dDeviceContext       = pd3dDeviceContext; 
    m_pd3dDevice              = pd3dDevice;
     
    m_GPULightEnvAlloc.Init(pd3dDevice);
    m_DynamicVB.Init(pd3dDevice, 4 * 1024 * 1024);

    m_DynamicVertexAllocCount = 0;
    m_DynamicVertexDrawnCount = 0;
    m_DynamicVertexOffset     = 0;
    m_DynamicVertexStride     = 0;
    m_DynamicVertexAlloc      = 0;

    XMVECTOR vMeshMin;
    XMVECTOR vMeshMax;

    m_vSceneAABBMin = g_vFLTMAX; 
    m_vSceneAABBMax = g_vFLTMIN;

    // Load the shaders
    ReloadShaders();

    // Calculate the AABB for the scene by iterating through all the meshes in the SDKMesh file.
    for( UINT i =0; i < pMesh->GetNumMeshes( ); ++i ) 
    {
        auto msh = pMesh->GetMesh( i );
        vMeshMin = XMVectorSet( msh->BoundingBoxCenter.x - msh->BoundingBoxExtents.x,
             msh->BoundingBoxCenter.y - msh->BoundingBoxExtents.y,
             msh->BoundingBoxCenter.z - msh->BoundingBoxExtents.z,
             1.0f );

        vMeshMax = XMVectorSet( msh->BoundingBoxCenter.x + msh->BoundingBoxExtents.x,
             msh->BoundingBoxCenter.y + msh->BoundingBoxExtents.y,
             msh->BoundingBoxCenter.z + msh->BoundingBoxExtents.z,
             1.0f );
        
        m_vSceneAABBMin = XMVectorMin( vMeshMin, m_vSceneAABBMin );
        m_vSceneAABBMax = XMVectorMax( vMeshMax, m_vSceneAABBMax );
    }

    float    fpadding = 0.0f;
    XMVECTOR padding  = XMLoadFloat(&fpadding);
    m_vSceneAABBMin   = XMVectorSubtract(m_vSceneAABBMin, padding);
    m_vSceneAABBMax   = XMVectorAdd(     m_vSceneAABBMax, padding);

    // Light vertex buffer
    {
      D3D11_BUFFER_DESC Desc = { 0 };
      Desc.Usage             = D3D11_USAGE_DEFAULT;
      Desc.BindFlags         = D3D11_BIND_VERTEX_BUFFER;
      Desc.ByteWidth         = sizeof( s_EllipsoidVertTable );

      D3D11_SUBRESOURCE_DATA  data = { 0 };
      data.pSysMem                 = s_EllipsoidVertTable;
      V_RETURN( pd3dDevice->CreateBuffer( &Desc, &data, &m_pLightVB ) );
      DXUT_SetDebugName( m_pLightVB, "m_pLightVB" );
    }

    {
      const D3D11_INPUT_ELEMENT_DESC layout_light[] =
      {
          { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 }
      };
    
      V_RETURN( pd3dDevice->CreateInputLayout(layout_light, ARRAYSIZE( layout_light ), 
                                              m_pvsRenderLight.m_ShaderBlob->GetBufferPointer(),
                                              m_pvsRenderLight.m_ShaderBlob->GetBufferSize(), 
                                              &m_pVertexLayoutLight ) );
      DXUT_SetDebugName( m_pVertexLayoutLight, "SceneManager light Layout" );
    }

    {
      const D3D11_INPUT_ELEMENT_DESC layout_mesh[] =
      {
          { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
          { "NORMAL",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
          { "TEXCOORD",  0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
      };
    
      V_RETURN( pd3dDevice->CreateInputLayout(layout_mesh, ARRAYSIZE( layout_mesh ), 
                                              m_pvsRenderScene.m_ShaderBlob->GetBufferPointer(),
                                              m_pvsRenderScene.m_ShaderBlob->GetBufferSize(), 
                                              &m_pVertexLayoutMesh ) );
      DXUT_SetDebugName( m_pVertexLayoutMesh, "SceneManager mesh layout" );
    }
     
    // Raster states  
    { 
      D3D11_RASTERIZER_DESC drd = 
      {
        D3D11_FILL_SOLID,       // D3D11_FILL_MODE FillMode;
        D3D11_CULL_BACK,        // D3D11_CULL_MODE CullMode;
        false,                  // BOOL FrontCounterClockwise;
        0,                      // INT DepthBias;
        0.f,                    // FLOAT DepthBiasClamp;
        0.f,                    // FLOAT SlopeScaledDepthBias;
        true,                   // BOOL DepthClipEnable;
        false,                  // BOOL ScissorEnable;
        false,                  // BOOL MultisampleEnable;
        false,                  // BOOL AntialiasedLineEnable;
      };

      drd.CullMode = D3D11_CULL_NONE;
      pd3dDevice->CreateRasterizerState( &drd, &m_prsCullNone );
      DXUT_SetDebugName( m_prsCullNone, "m_prsCullNone" );
      
      drd.CullMode = D3D11_CULL_BACK;
      pd3dDevice->CreateRasterizerState( &drd, &m_prsCullBackFaces );
      DXUT_SetDebugName( m_prsCullBackFaces, "m_prsCullBackFaces" );

      drd.CullMode = D3D11_CULL_FRONT;
      pd3dDevice->CreateRasterizerState( &drd, &m_prsCullFrontFaces );
      DXUT_SetDebugName( m_prsCullFrontFaces, "m_prsCullFrontFaces" );

      drd.CullMode             = D3D11_CULL_NONE;
      // Setting the slope scale depth bias greatly decreases surface acne and incorrect self shadowing.
      drd.SlopeScaledDepthBias = 1.0;
      pd3dDevice->CreateRasterizerState( &drd, &m_prsShadow );
      DXUT_SetDebugName( m_prsShadow, "m_prsShadow" );
    }
    
    D3D11_BUFFER_DESC Desc = { 0 };
    Desc.Usage          = D3D11_USAGE_DYNAMIC;
    Desc.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    // Light Instances
    {
      Desc.ByteWidth = sizeof( LightInstancesCB );
      V_RETURN( pd3dDevice->CreateBuffer( &Desc, nullptr, &m_pcbLightInstancesCB ) );
      DXUT_SetDebugName( m_pcbLightInstancesCB, "m_pcbLightInstancesCB" );
    }

    // Frame
    {
      Desc.ByteWidth = sizeof( FrameCB );
      V_RETURN( pd3dDevice->CreateBuffer( &Desc, nullptr, &m_pcbFrameCB ) );
      DXUT_SetDebugName( m_pcbFrameCB, "FrameCB" );
    }

    // Shadow
    {
      Desc.ByteWidth = sizeof( ShadowDataCB );
      V_RETURN( pd3dDevice->CreateBuffer( &Desc, nullptr, &m_pcbShadowCB ) );
      DXUT_SetDebugName( m_pcbShadowCB, "ShadowDataCB" );
    }

    // Simple
    {
      Desc.ByteWidth = sizeof( SimpleCB );
      V_RETURN( pd3dDevice->CreateBuffer( &Desc, nullptr, &m_pcbSimpleCB ) );
      DXUT_SetDebugName( m_pcbSimpleCB, "SimpleCB" );
    }
 
    // Blend States
    {
      {
        D3D11_BLEND_DESC desc = { 0 };

        desc.RenderTarget[0].BlendEnable           = false;                              // BOOL BlendEnable;
        desc.RenderTarget[0].SrcBlend              = D3D11_BLEND_ONE;                    // D3D11_BLEND SrcBlend;
        desc.RenderTarget[0].DestBlend             = D3D11_BLEND_ZERO;                   // D3D11_BLEND DestBlend;
        desc.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;                 // D3D11_BLEND_OP BlendOp;
        desc.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;                    // D3D11_BLEND SrcBlendAlpha;
        desc.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ZERO;                   // D3D11_BLEND DestBlendAlpha;
        desc.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;                 // D3D11_BLEND_OP BlendOpAlpha;
        desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;       // UINT8 RenderTargetWriteMask;
     
        V_RETURN( pd3dDevice->CreateBlendState(&desc, &m_pbsNone) );
        DXUT_SetDebugName( m_pbsNone, "BlendState m_pbsNone" );
      }

      {
        D3D11_BLEND_DESC desc = { 0 };

        desc.RenderTarget[0].BlendEnable           = false;                              // BOOL BlendEnable;
        desc.RenderTarget[0].SrcBlend              = D3D11_BLEND_ONE;                    // D3D11_BLEND SrcBlend;
        desc.RenderTarget[0].DestBlend             = D3D11_BLEND_ZERO;                   // D3D11_BLEND DestBlend;
        desc.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;                 // D3D11_BLEND_OP BlendOp;
        desc.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;                    // D3D11_BLEND SrcBlendAlpha;
        desc.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ZERO;                   // D3D11_BLEND DestBlendAlpha;
        desc.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;                 // D3D11_BLEND_OP BlendOpAlpha;
        desc.RenderTarget[0].RenderTargetWriteMask = 0;                                  // UINT8 RenderTargetWriteMask;
     
        V_RETURN( pd3dDevice->CreateBlendState(&desc, &m_pbsDisableRGBA) );
        DXUT_SetDebugName( m_pbsDisableRGBA, "BlendState m_pbsDisableRGBA" );
      }

      {
        D3D11_BLEND_DESC desc = { 0 };

        desc.RenderTarget[0].BlendEnable           = true;                          // BOOL BlendEnable;
        desc.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;         // D3D11_BLEND SrcBlend;
        desc.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;     // D3D11_BLEND DestBlend;
        desc.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;            // D3D11_BLEND_OP BlendOp;
        desc.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_SRC_ALPHA;         // D3D11_BLEND SrcBlendAlpha;
        desc.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_INV_SRC_ALPHA;     // D3D11_BLEND DestBlendAlpha;
        desc.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;            // D3D11_BLEND_OP BlendOpAlpha;
        desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;  // UINT8 RenderTargetWriteMask;

        V_RETURN( pd3dDevice->CreateBlendState(&desc, &m_pbsAlpha) );
        DXUT_SetDebugName( m_pbsAlpha, "BlendState m_pbsAlpha" );
      }
    }


    // Samplers
    {
      D3D11_SAMPLER_DESC SamDescLin=
      {
        D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_WRAP,
        D3D11_TEXTURE_ADDRESS_WRAP,
        D3D11_TEXTURE_ADDRESS_WRAP,
        0.0f,
        1,
        D3D11_COMPARISON_ALWAYS,
        0.0,0.0,0.0,0.0,//FLOAT BorderColor[ 4 ];
        0,//FLOAT MinLOD;
        D3D11_FLOAT32_MAX
      };
      V_RETURN( m_pd3dDevice->CreateSamplerState( &SamDescLin, &m_pSamLinear ) );
      DXUT_SetDebugName( m_pSamLinear, "Sampler Linear" );

      D3D11_SAMPLER_DESC SamDescPt=
      {
        D3D11_FILTER_MIN_MAG_MIP_POINT,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        0.0f,
        0,
        D3D11_COMPARISON_ALWAYS,
        0.0,0.0,0.0,0.0,//FLOAT BorderColor[ 4 ];
        0,//FLOAT MinLOD;
        D3D11_FLOAT32_MAX
      };
      V_RETURN( m_pd3dDevice->CreateSamplerState( &SamDescPt, &m_pSamPoint ) );
      DXUT_SetDebugName( m_pSamPoint, "Sampler Point" );

      D3D11_SAMPLER_DESC SamDescShad = 
      {
        D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,// D3D11_FILTER Filter;
        D3D11_TEXTURE_ADDRESS_BORDER, //D3D11_TEXTURE_ADDRESS_MODE AddressU;
        D3D11_TEXTURE_ADDRESS_BORDER, //D3D11_TEXTURE_ADDRESS_MODE AddressV;
        D3D11_TEXTURE_ADDRESS_BORDER, //D3D11_TEXTURE_ADDRESS_MODE AddressW;
        0,//FLOAT MipLODBias;
        0,//UINT MaxAnisotropy;
        D3D11_COMPARISON_LESS , //D3D11_COMPARISON_FUNC ComparisonFunc;
        0.0,0.0,0.0,0.0,//FLOAT BorderColor[ 4 ];
        0,//FLOAT MinLOD;
        0//FLOAT MaxLOD;   
      };

      V_RETURN( m_pd3dDevice->CreateSamplerState( &SamDescShad, &m_pSamShadowPCF ) );
      DXUT_SetDebugName( m_pSamShadowPCF, "Sampler Shadow PCF" );
    }

    // Depth states
    {
      D3D11_DEPTH_STENCIL_DESC depth_stencil_desc =
      {
        true,                               // BOOL DepthEnable;
        D3D11_DEPTH_WRITE_MASK_ALL,         // D3D11_DEPTH_WRITE_MASK DepthWriteMask;
        D3D11_COMPARISON_LESS_EQUAL,        // D3D11_COMPARISON_FUNC DepthFunc;

        false,                              // BOOL StencilEnable;
        D3D11_DEFAULT_STENCIL_READ_MASK,    // UINT8 StencilReadMask;
        D3D11_DEFAULT_STENCIL_WRITE_MASK,   // UINT8 StencilWriteMask;

        D3D11_STENCIL_OP_KEEP,              // D3D11_DEPTH_STENCILOP_DESC FrontFace;  D3D11_STENCIL_OP StencilFailOp;
        D3D11_STENCIL_OP_KEEP,              //                                        D3D11_STENCIL_OP StencilDepthFailOp;
        D3D11_STENCIL_OP_KEEP,              //                                        D3D11_STENCIL_OP StencilPassOp;
        D3D11_COMPARISON_ALWAYS,            //                                        D3D11_COMPARISON_FUNC StencilFunc;

        D3D11_STENCIL_OP_KEEP,              // D3D11_DEPTH_STENCILOP_DESC BackFace;   D3D11_STENCIL_OP StencilFailOp;
        D3D11_STENCIL_OP_KEEP,              //                                        D3D11_STENCIL_OP StencilDepthFailOp;
        D3D11_STENCIL_OP_KEEP,              //                                        D3D11_STENCIL_OP StencilPassOp;
        D3D11_COMPARISON_ALWAYS,            //                                        D3D11_COMPARISON_FUNC StencilFunc;
      };

      V_RETURN( m_pd3dDevice->CreateDepthStencilState( &depth_stencil_desc, &m_pdsDefault ) );
      DXUT_SetDebugName( m_pdsDefault, "DepthStencil Default" );

      depth_stencil_desc.DepthWriteMask       = D3D11_DEPTH_WRITE_MASK_ZERO;

      V_RETURN( m_pd3dDevice->CreateDepthStencilState( &depth_stencil_desc, &m_pdsNoWrite ) );
      DXUT_SetDebugName( m_pdsNoWrite, "DepthStencil No Write" );
    }

    // Allocate shadow target
    {
      for(INT index=0; index < CASCADE_COUNT_FLAG; ++index ) 
      { 
        m_RenderVP[index].Height   = (FLOAT)CASCADE_BUFFER_SIZE;
        m_RenderVP[index].Width    = (FLOAT)CASCADE_BUFFER_SIZE;
        m_RenderVP[index].MaxDepth = 1.0f;
        m_RenderVP[index].MinDepth = 0.0f;
        m_RenderVP[index].TopLeftX = (FLOAT)(CASCADE_BUFFER_SIZE * index );
        m_RenderVP[index].TopLeftY = 0;
      }

      m_RenderOneTileVP.Height     = (FLOAT)CASCADE_BUFFER_SIZE;
      m_RenderOneTileVP.Width      = (FLOAT)CASCADE_BUFFER_SIZE;
      m_RenderOneTileVP.MaxDepth   = 1.0f;
      m_RenderOneTileVP.MinDepth   = 0.0f;
      m_RenderOneTileVP.TopLeftX   = 0.0f;
      m_RenderOneTileVP.TopLeftY   = 0.0f;
       
      m_CascadedShadowMapRT.Init(CASCADE_BUFFER_SIZE * CASCADE_COUNT_FLAG,
                                 CASCADE_BUFFER_SIZE,
                                 DXGI_FORMAT_UNKNOWN,
                                 DXGI_FORMAT_D16_UNORM);
    }

    // Clear the viewport
    ZeroMemory(&m_MainVP, sizeof(m_MainVP));

    // Allocate screen resources
    OnResize(width, height);

    // Done
    return hr;
}

//--------------------------------------------------------------------------------------
HRESULT   SceneManager::ReloadShaders()
{
  HRESULT hr = S_OK;

  m_pvsRenderSimple.Release();
  m_pvsRenderScene.Release();      
  m_pvsRenderLight.Release();      
  m_pvsRender2D.Release();         
  
  m_ppsComposite.Release();        
  m_ppsInsertLightNoCulling.Release(); 
  m_ppsInsertLightBackFace.Release();  
  m_ppsDebugLight.Release();       
  m_ppsClearLLL.Release();         
  m_ppsTexture.Release();          
  m_ppsGBuffer.Release();  
  m_ppsLit3D.Release();     
          
  V_RETURN( CreateVertexShader(&m_pvsRenderSimple,          m_pd3dDevice, L"Shaders.hlsl",  "VSMainSimple"));
  V_RETURN( CreateVertexShader(&m_pvsRenderScene,           m_pd3dDevice, L"Shaders.hlsl",  "VSMainScene" ));
  V_RETURN( CreateVertexShader(&m_pvsRenderLight,           m_pd3dDevice, L"Shaders.hlsl",  "VSMainLight" ));
  V_RETURN( CreateVertexShader(&m_pvsRender2D,              m_pd3dDevice, L"Shaders.hlsl",  "VSMain2D"    ));
                                                            
  V_RETURN( CreatePixelShader( &m_ppsLit3D,                 m_pd3dDevice, L"Shaders.hlsl",  "PSLit3D"                ));
  V_RETURN( CreatePixelShader( &m_ppsComposite,             m_pd3dDevice, L"Shaders.hlsl",  "PSComposite"            ));
  V_RETURN( CreatePixelShader( &m_ppsInsertLightNoCulling,  m_pd3dDevice, L"Shaders.hlsl",  "PSInsertLightNoCulling" ));
  V_RETURN( CreatePixelShader( &m_ppsInsertLightBackFace,   m_pd3dDevice, L"Shaders.hlsl",  "PSInsertLightBackFace"  ));
  V_RETURN( CreatePixelShader( &m_ppsDebugLight,            m_pd3dDevice, L"Shaders.hlsl",  "PSDebugLight"           ));
  V_RETURN( CreatePixelShader( &m_ppsClearLLL,              m_pd3dDevice, L"Shaders.hlsl",  "PSClearLLLEighth"       ));
  V_RETURN( CreatePixelShader( &m_ppsTexture,               m_pd3dDevice, L"Shaders.hlsl",  "PSTexture"              ));
  V_RETURN( CreatePixelShader( &m_ppsGBuffer,               m_pd3dDevice, L"Shaders.hlsl",  "PSGBuffer"              ));

  return hr;
}

//--------------------------------------------------------------------------------------
void   SceneManager::DrawEllipsoidLightShells(int inst_count)
{
  uint32_t v_stride = sizeof(float4);
  uint32_t v_offset = 0;

  // Bind the static vertex buffer
  m_pd3dDeviceContext->IASetVertexBuffers( 0, 1, &m_pLightVB, &v_stride, &v_offset );

  // Draw the geometry
  m_pd3dDeviceContext->DrawInstanced(ARRAYSIZE(s_EllipsoidVertTable), inst_count, 0, 0);
}

//--------------------------------------------------------------------------------------
void   SceneManager::ReDrawEllipsoidLightShell()
{
  // Draw the geometry
  m_pd3dDeviceContext->Draw(ARRAYSIZE(s_EllipsoidVertTable), 0);
}

//--------------------------------------------------------------------------------------------------
void*   SceneManager::ScratchAlloc(uint32_t size)
{
  size_t alignment         =  16;
  size_t alignment_inc     = (alignment - 1);
  size_t alignment_mask    = ~alignment_inc;

  size_t base_misalignment = (size_t)m_ScratchBase & alignment_inc;

  // align the current offset
  size_t aligned_offset    = ((m_ScratchOffset + base_misalignment + alignment_inc) & alignment_mask) - base_misalignment;

  // check to make sure we have room
  if ( aligned_offset + size > m_ScratchSize )
  {
    return nullptr;
  }

  m_ScratchOffset   = (uint32_t)(aligned_offset + size);
 
  void* ptr         = (void*)((uint8_t*)m_ScratchBase + aligned_offset);

  return ptr;
}

//--------------------------------------------------------------------------------------
HRESULT SceneManager::ReleaseResources() 
{
  free(m_ScratchBase);  
  m_ScratchBase             = nullptr;
  m_ScratchSize             = 0; 
  m_ScratchOffset           = 0;

  SAFE_RELEASE( m_pSamShadowPCF );
  SAFE_RELEASE( m_pSamLinear );
  SAFE_RELEASE( m_pSamPoint );
  
  SAFE_RELEASE( m_pVertexLayoutLight ); 
  SAFE_RELEASE( m_pVertexLayoutMesh  ); 
  
  SAFE_RELEASE( m_pdsDefault );
  SAFE_RELEASE( m_pdsNoWrite );
  
  SAFE_RELEASE( m_pcbLightInstancesCB);
  SAFE_RELEASE( m_pcbSimpleCB        );
  SAFE_RELEASE( m_pcbShadowCB        );
  SAFE_RELEASE( m_pcbFrameCB         );
 
  SAFE_RELEASE( m_pLightVB    );

  SAFE_RELEASE( m_prsCullFrontFaces );
  SAFE_RELEASE( m_prsCullBackFaces  );
  SAFE_RELEASE( m_prsCullNone       );
  SAFE_RELEASE( m_prsShadow         );
  
  SAFE_RELEASE( m_pbsDisableRGBA );
  SAFE_RELEASE( m_pbsAlpha       )
  SAFE_RELEASE( m_pbsNone        );
  
  m_CascadedShadowMapRT.Release();
  m_GBufferRT.Release();
  
  m_LLLTarget.Release();
  
  m_pvsRenderSimple.Release();
  m_pvsRenderLight.Release();
  m_pvsRenderScene.Release();
  m_pvsRender2D.Release();
  
  m_ppsComposite.Release();
  m_ppsLit3D.Release();

  m_ppsInsertLightNoCulling.Release(); 
  m_ppsInsertLightBackFace.Release(); 

  m_ppsDebugLight.Release();
  m_ppsClearLLL.Release();
  m_ppsGBuffer.Release();
  m_ppsTexture.Release();
  
  m_GPULightEnvAlloc.Destroy();
  m_DynamicVB.Destroy();

  return S_OK;
}


//--------------------------------------------------------------------------------------
void SceneManager::SetSampler(uint32_t slot,  ID3D11SamplerState* sam)
{
  m_pd3dDeviceContext->PSSetSamplers( slot, 1, &sam );
}

//--------------------------------------------------------------------------------------
void SceneManager::ClearSamplers(uint32_t slot, uint32_t count)
{
  if(count)
  {    
    m_pd3dDeviceContext->PSSetSamplers( slot, count, (ID3D11SamplerState* *)g_ZeroMemory ); 
  }
}

//--------------------------------------------------------------------------------------
void SceneManager::SetResourceView(uint32_t slot,  ID3D11ShaderResourceView* view)
{
  m_pd3dDeviceContext->PSSetShaderResources( slot, 1, &view );
}

//--------------------------------------------------------------------------------------
void SceneManager::ClearResourceViews(uint32_t slot, uint32_t count)
{
  if(count)
  {    
    m_pd3dDeviceContext->PSSetShaderResources( slot, count, (ID3D11ShaderResourceView* *)g_ZeroMemory ); 
  }
}

//--------------------------------------------------------------------------------------------------
HRESULT DynamicD3DBuffer::Init(ID3D11Device* pd3dDevice, uint32_t max_size)
{
  Clear();

  m_Size                  = max_size;

  D3D11_BUFFER_DESC bd    = { 0 };

  bd.Usage                = D3D11_USAGE_DYNAMIC;
  bd.ByteWidth            = m_Size;
  bd.BindFlags            = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_INDEX_BUFFER;
  bd.CPUAccessFlags       = D3D11_CPU_ACCESS_WRITE;
  bd.MiscFlags            = 0;
  bd.StructureByteStride  = 0;
   
  return pd3dDevice->CreateBuffer( &bd, nullptr, &m_D3DBuffer );
}


//--------------------------------------------------------------------------------------------------
void* DynamicD3DBuffer::Alloc( size_t size, uint32_t& offset )
{
  uint32_t aligned_size = (uint32_t)ALIGN_16(size);

  // Handle the case that we passed the end of the active buffer
  if (m_CurrentPos + aligned_size > m_Size)
  {
    return 0;
  }

  if ( m_BaseAddress )
  {
    offset        = m_CurrentPos;
    m_CurrentPos += aligned_size;
    return (void*)((uintptr_t)m_BaseAddress + offset);
  }

  return NULL;
}

//--------------------------------------------------------------------------------------------------
void DynamicD3DBuffer::BeginFrame(ID3D11DeviceContext* pd3dDeviceContext)
{  
  D3D11_MAPPED_SUBRESOURCE res;
  if ( pd3dDeviceContext->Map(m_D3DBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &res) == S_OK )
  {
    pd3dDeviceContext->Unmap(m_D3DBuffer, 0);
    m_BaseAddress = res.pData;
  } 
  m_CurrentPos = 0;
}
 
//--------------------------------------------------------------------------------------------------
bool  GPULightEnvAlloc::Init(ID3D11Device* pd3dDevice)
{ 
  int       total_alloc_size        = sizeof(GPULightEnv) * (MAX_LLL_LIGHTS + 32);
     
  // Create the Vertex buffer
  {
    D3D11_BUFFER_DESC   buffer_desc = { 0 };

    buffer_desc.Usage               = D3D11_USAGE_DYNAMIC;
    buffer_desc.CPUAccessFlags      = D3D11_CPU_ACCESS_WRITE;  
    buffer_desc.ByteWidth           = total_alloc_size;
    buffer_desc.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    buffer_desc.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    buffer_desc.StructureByteStride = sizeof(GPULightEnv);

    HRESULT       hr                = S_OK;

    // Create the buffer
    hr                              = pd3dDevice->CreateBuffer(&buffer_desc, NULL, &m_StructuredBuffer);

    // Validate the buffer
    if(hr != S_OK)
    {
      assert(!"GPULightEnvAlloc::Init: Failed to create the D3DBuffer!");
      return false;
    }
     
    D3D11_BUFFER_DESC descBuf;
    m_StructuredBuffer->GetDesc( &descBuf );

    D3D11_SHADER_RESOURCE_VIEW_DESC descView;
    memset( &descView, 0x00, sizeof(descView) );
    descView.ViewDimension         = D3D11_SRV_DIMENSION_BUFFEREX;
    descView.BufferEx.FirstElement = 0;

    descView.Format               = DXGI_FORMAT_UNKNOWN;
    descView.BufferEx.NumElements = descBuf.ByteWidth/descBuf.StructureByteStride;
    hr                            = pd3dDevice->CreateShaderResourceView( m_StructuredBuffer,  &descView, &m_StructuredBufferViewer );

     // Validate the buffer
     if(hr != S_OK)
     {
       assert(!"Failed to create the resource view");
       return false;
     }
  }
  
  m_FrameMemMax         = total_alloc_size;

  // Clear member variables
  m_FrameMem            = NULL;
  m_FrameMemOffset      = 0;  
  m_ReflStartIndex      = 0;
  m_ReflEndIndex        = 0;

  // done
  return true;
}


//--------------------------------------------------------------------------------------------------
GPULightEnv* GPULightEnvAlloc::AllocateReflectionVolumes(uint32_t count)
{
  uint32_t      start_idx  = GetAllocCount();
  uint32_t      end_idx    = start_idx + count;

  GPULightEnv*  alloc      = Allocate(count);

  // Validate
  if(alloc == NULL)
  {
    return NULL;
  }

  m_ReflStartIndex = start_idx;
  m_ReflEndIndex   = end_idx;

  // Done
  return alloc;
}

//--------------------------------------------------------------------------------------------------
GPULightEnv* GPULightEnvAlloc::Allocate(uint32_t count)
{
  assert(m_FrameMemOffset <= m_FrameMemMax);

  GPULightEnv* env        = NULL;
  uint32_t     alloc_size = count * sizeof(GPULightEnv);

  if((m_FrameMem != NULL) && (m_FrameMemOffset + alloc_size <= m_FrameMemMax))
  {
    env               = (GPULightEnv*)(m_FrameMem + m_FrameMemOffset);
    m_FrameMemOffset += alloc_size;
  }

  // Done
  return env;
}

//-------------------------------------------------------------------------------------------------- 
void  GPULightEnvAlloc::BeginFrame(ID3D11DeviceContext*    pd3dDeviceContext)
{
  m_FrameMemOffset = 0;  
  m_ReflStartIndex = 0;
  m_ReflEndIndex   = 0;

  {
    D3D11_MAPPED_SUBRESOURCE res; 
    pd3dDeviceContext->Map(m_StructuredBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &res);
    m_FrameMem     = (uint8_t*)res.pData;  
    pd3dDeviceContext->Unmap(m_StructuredBuffer, 0);
  }
}

//--------------------------------------------------------------------------------------------------
void  GPULightEnvAlloc::Destroy()
{
  SAFE_RELEASE(m_StructuredBufferViewer); 
  SAFE_RELEASE(m_StructuredBuffer); 

  m_StructuredBufferViewer = NULL; 
  m_StructuredBuffer       = NULL; 
}

 
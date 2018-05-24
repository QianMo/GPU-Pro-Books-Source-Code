/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");// you may not use this file except in compliance with the License.// You may obtain a copy of the License at//// http://www.apache.org/licenses/LICENSE-2.0//// Unless required by applicable law or agreed to in writing, software// distributed under the License is distributed on an "AS IS" BASIS,// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.// See the License for the specific language governing permissions and// limitations under the License.
//
// Modified by StephanieB5 to remove dependencies on DirectX SDK in 2017
//
/////////////////////////////////////////////////////////////////////////////////////////////

#include "ColorUtil.h"

using namespace DirectX;

XMVECTOR HueToRGB(float hue)
{
    float intPart;
    float fracPart = modff(hue * 6.0f, &intPart);
    int region = static_cast<int>(intPart);
    
    switch (region) {
    case 0: return  XMVectorSet(1.0f, fracPart, 0.0f, 0.f);
    case 1: return  XMVectorSet(1.0f - fracPart, 1.0f, 0.0f, 0.f);
    case 2: return  XMVectorSet(0.0f, 1.0f, fracPart, 0.f);
    case 3: return  XMVectorSet(0.0f, 1.0f - fracPart, 1.0f, 0.f);
    case 4: return  XMVectorSet(fracPart, 0.0f, 1.0f, 0.f);
    case 5: return  XMVectorSet(1.0f, 0.0f, 1.0f - fracPart, 0.f);
    };

    return  XMVectorSet(0.0f, 0.0f, 0.0f, 0.f);
}

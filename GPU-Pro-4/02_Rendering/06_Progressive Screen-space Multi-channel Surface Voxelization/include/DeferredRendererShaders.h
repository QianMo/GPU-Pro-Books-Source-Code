//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  EaZD Deferred Renderer                                                  //
//  Georgios Papaioannou, 2009-10                                             //
//                                                                          //
//  This is a free deferred renderer. The library and the source            //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

// Multiple render target core shaders
#include "shaders/DeferredRendererShader_MRT.h"
#include "shaders/DeferredRendererShader_ClearMRT.h"
#include "shaders/DeferredRendererShader_Transparency.h"
#include "shaders/DeferredRendererShader_ShadowMap.h"

// Deferred rendering core shaders
#include "shaders/DeferredRendererShader_Illumination.h"
#include "shaders/DeferredRendererShader_FrameBuffer.h"
#include "shaders/DeferredRendererShader_Glow.h"
#include "shaders/DeferredRendererShader_PostProcess.h"

// Shaders for Viewing results
#include "shaders/DeferredRendererShader_ViewDepthBuffer.h"
#include "shaders/DeferredRendererShader_ViewPhotonMap.h"

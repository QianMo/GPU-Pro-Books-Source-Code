/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH_GENERATION_DX11
#define BE_SCENE_MESH_GENERATION_DX11

#include "../beScene.h"
#include "../beMeshGeneration.h"
#include <D3D11.h>
#include <beCore/beExchangeContainers.h>

namespace beScene
{
namespace DX11
{

/// Computes a vertex description from the given mesh generation flags.
beCore::Exchange::vector_t<D3D11_INPUT_ELEMENT_DESC>::t ComputeVertexDesc(uint4 meshGenFlags);

/// Computes the index format from the given vertex count.
DXGI_FORMAT ComputeIndexFormat(uint4 vertexCount, uint4 meshGenFlags);

} // namespace
} // namespace

#endif
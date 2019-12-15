#pragma once

#include "../../Core/Rendering/Platform11/Platform11.h"
#include "../../Core/Math/Math.h"
#include "../Shaders/AtlasQuads.inc"

namespace AtlasQuads
{
    struct SQuad
    {
        UVec4 m_pos;

        static const SQuad Get(
            int dstRectW, int dstRectH,
            int dstX, int dstY, int dstW, int dstH )
        {
            SQuad q;

            q.m_pos.x = float(dstRectW) / float(dstW);
            q.m_pos.y = float(dstRectH) / float(dstH);
            q.m_pos.z = q.m_pos.x + 2.0f*float(dstX)/float(dstW) - 1.0f;
            q.m_pos.w = 1.0f - q.m_pos.y - 2.0f*float(dstY)/float(dstH);

            return q;
        }
    };

    struct SFillQuad : public SQuad
    {
        UVec4 m_misc;

        static const SFillQuad Get(
            const Vec4& miscParams,
            int dstRectW, int dstRectH,
            int dstX, int dstY, int dstW, int dstH )
        {
            SFillQuad q;

            static_cast< SQuad& >( q ) = SQuad::Get( dstRectW, dstRectH, dstX, dstY, dstW, dstH );

            q.m_misc = miscParams;

            return q;
        }
    };

    struct SCopyQuad : public SFillQuad
    {
        UVec4 m_texCoord;

        static const SCopyQuad Get(
            const Vec4& miscParams,
            int dstRectW, int dstRectH,
            int dstX, int dstY, int dstW, int dstH, 
            int srcRectW, int srcRectH,
            int srcX, int srcY, int srcW, int srcH)
        {
            SCopyQuad q;

            static_cast< SFillQuad& >( q ) = SFillQuad::Get( miscParams, dstRectW, dstRectH, dstX, dstY, dstW, dstH );

            // Align with pixel center @ (0.5, 0.5).
            q.m_pos.z += 1.0f/float(dstW);
            q.m_pos.w -= 1.0f/float(dstH);

            q.m_texCoord.x = float(srcRectW) / float(srcW);
            q.m_texCoord.y = float(srcRectH) / float(srcH);
            q.m_texCoord.z = ( float(srcX) + 0.5f ) / float(srcW);
            q.m_texCoord.w = ( float(srcY) + 0.5f ) / float(srcH);

            return q;
        }
    };

    template< class T >
    inline void Draw( unsigned int numQuads, const T* pQuads, DeviceContext11& dc = Platform::GetImmediateContext() )
    {
        static const unsigned int registersPerQuad = sizeof( T ) / sizeof( UVec4 );
        static const unsigned int maxQuads = QUADS_ARRAY_REGS / registersPerQuad;

        dc.UnbindVertexBuffer(0);
        dc.UnbindIndexBuffer();
        dc.SetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

        for( unsigned int i = 0; i < numQuads; )
        {
            unsigned int toRender = std::min( numQuads - i, maxQuads );

            ID3D11Buffer* pQuadsData = dc.GetConstantBuffers().Allocate( toRender * registersPerQuad * sizeof( UVec4 ), pQuads + i, dc.DoNotFlushToDevice() );

            dc.VSSetConstantBuffer( 3, pQuadsData );

            dc.FlushToDevice()->Draw( toRender * 6, 0 );

            dc.GetConstantBuffers().Free( pQuadsData );

            i += toRender;
        }
    }

} // namespace AtlasQuads

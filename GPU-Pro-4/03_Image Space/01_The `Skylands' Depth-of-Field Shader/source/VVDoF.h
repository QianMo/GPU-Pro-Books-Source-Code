#ifndef VVDoF_h
#define VVDoF_h

#include <G3D/G3DAll.h>

/** 
  \brief
 */
class VVDoF : public ReferenceCountedObject {
public:
    
    enum DebugOption {
        NONE,
        SHOW_COC,
        SHOW_REGION,
        SHOW_NEAR,
        SHOW_BLURRY,
        SHOW_INPUT,
        SHOW_MID_AND_FAR,
        SHOW_SIGNED_COC
    };

protected:

    VVDoF();

    /** Color in RGB, circle of confusion and 'near field' bit in A. 
        Precision determined by the input, RGB8, RGB16F, or RGB32F.

        The A channel values are always written with only 8 bits of
        effective precision.

        The radius (A channel) values are scaled and biased to [0, 1].
        Unpack them to pixel radii with:

        \code
        r = ((a * 2) - 1) * maxRadius
        \endcode

        Where maxRadius the larger of the maximum near and far field
        blurs.  The decoded radius is negative in the far field (the packed
        alpha channel should look like a head lamp on a dark night, with
        nearby objects bright, the focus field gray, and the distance black).
    */
    shared_ptr<Texture>         m_packedBuffer;
    shared_ptr<Framebuffer>     m_packedFramebuffer;
    shared_ptr<Shader>          m_artistCoCShader;
    shared_ptr<Shader>          m_physicalCoCShader;

    shared_ptr<Framebuffer>     m_horizontalFramebuffer;
    shared_ptr<Shader>          m_horizontalShader;
    shared_ptr<Texture>         m_tempNearBuffer;
    shared_ptr<Texture>         m_tempBlurBuffer;

    shared_ptr<Framebuffer>     m_verticalFramebuffer;
    shared_ptr<Shader>          m_verticalShader;
    shared_ptr<Texture>         m_nearBuffer;
    shared_ptr<Texture>         m_blurBuffer;

    shared_ptr<Shader>          m_compositeShader;

    /** Allocates and resizes buffers */
    void resizeBuffers(shared_ptr<Texture> target);

    /** Writes m_packedBuffer */
    void computeCoC(RenderDevice* rd, const shared_ptr<Texture>& color, const shared_ptr<Texture>& depth, const shared_ptr<Camera>& camera);

    void blurPass(RenderDevice* rd, const shared_ptr<Texture>& blurInput, const shared_ptr<Texture>& nearInput,
        const shared_ptr<Framebuffer>& output, const shared_ptr<Shader>& shader, const shared_ptr<Camera>& camera);

    /**
       Writes to the currently-bound framebuffer.
     */
    void composite(RenderDevice* rd, shared_ptr<Texture> packedBuffer, shared_ptr<Texture> blurBuffer, shared_ptr<Texture> nearBuffer, DebugOption d);

public:
    
    void reloadShaders();


    /** \brief Constructs an empty VVDoF. */
    static shared_ptr<VVDoF> create();

    /** Applies depth of field blur to supplied images and renders to
        the currently-bound framebuffer.  The current framebuffer may
        have the \a color and \a depth values bound to it.

        Reads depth reconstruction and circle of confusion parameters
        from \a camera.
    */
    void apply(RenderDevice* rd, shared_ptr<Texture> color, shared_ptr<Texture> depth, const shared_ptr<Camera>& camera, DebugOption debugOption = NONE);
};

#endif

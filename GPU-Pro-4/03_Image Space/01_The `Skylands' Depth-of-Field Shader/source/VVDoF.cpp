/**
 \file   VVDoF.cpp
 \author Morgan McGuire

TODO:
  - G3D::DepthOfField with trival lerp
  
  - Perf Optimization: Reduce to half-size on passes 
    - (test first with post-shrink)
*/
#include "VVDoF.h"

VVDoF::VVDoF() {
    // Intentionally empty
}


shared_ptr<VVDoF> VVDoF::create() {
    return shared_ptr<VVDoF>(new VVDoF());
}


void VVDoF::reloadShaders() {
    m_artistCoCShader = Shader::fromFiles("", System::findDataFile("DepthOfField/DepthOfField_artistCoC.pix"));
    m_artistCoCShader->setPreserveState(false);

    m_physicalCoCShader = Shader::fromFiles("", System::findDataFile("DepthOfField/DepthOfField_physicalCoC.pix"));
    m_physicalCoCShader->setPreserveState(false);

    m_horizontalShader = Shader::fromFiles("", "VVDoF_horizontal.pix");
    m_horizontalShader->setPreserveState(false);

    m_verticalShader = Shader::fromFiles("", "VVDoF_vertical.pix");
    m_verticalShader->setPreserveState(false);

    m_compositeShader = Shader::fromFiles("", "VVDoF_composite.pix");
    m_compositeShader->setPreserveState(false);
}


void VVDoF::apply
(RenderDevice*                  rd, 
 shared_ptr<Texture>            color,
 shared_ptr<Texture>            depth, 
 const shared_ptr<Camera>&      camera,
 DebugOption                    debugOption) {
    
    alwaysAssertM(notNull(color), "Color buffer may not be NULL");
    alwaysAssertM(notNull(depth), "Depth buffer may not be NULL");

    if (isNull(m_artistCoCShader)) {
        reloadShaders();
    }
    resizeBuffers(color);

    computeCoC(rd, color, depth, camera);
    blurPass(rd, m_packedBuffer, m_packedBuffer, m_horizontalFramebuffer, m_horizontalShader, camera);
    blurPass(rd, m_tempBlurBuffer, m_tempNearBuffer, m_verticalFramebuffer, m_verticalShader, camera);
    composite(rd, m_packedBuffer, m_blurBuffer, m_nearBuffer, debugOption);
}


/** Limit the maximum radius allowed for physical blur to 1% of the viewport or 12 pixels */
static float maxPhysicalBlurRadius(const Rect2D& viewport) {
    return max(viewport.width() / 100.0f, 12.0f);
}


void VVDoF::computeCoC
(RenderDevice*                 rd, 
 const shared_ptr<Texture>&    color, 
 const shared_ptr<Texture>&    depth, 
 const shared_ptr<Camera>&     camera) {

    rd->push2D(m_packedFramebuffer); {
        rd->clear();
        const double z_f    = camera->farPlaneZ();
        const double z_n    = camera->nearPlaneZ();
        
        const Vector3& clipInfo = 
            (z_f == -inf()) ? 
            Vector3(float(z_n), -1.0f, 1.0f) : 
            Vector3(float(z_n * z_f),  float(z_n - z_f),  float(z_f));
        
        shared_ptr<Shader> shader;
        if (camera->depthOfFieldSettings().model() == DepthOfFieldModel::ARTIST) {
            shader = m_artistCoCShader;
        } else {
            shader = m_physicalCoCShader;
        }

        Shader::ArgList& args = shader->args;

        args.set("color",     color);
        args.set("depth",     depth);
        args.set("clipInfo",  clipInfo);
        
        if (camera->depthOfFieldSettings().model() == DepthOfFieldModel::ARTIST) {

            args.set("nearBlurryPlaneZ", camera->depthOfFieldSettings().nearBlurryPlaneZ());
            args.set("nearSharpPlaneZ",  camera->depthOfFieldSettings().nearSharpPlaneZ());
            args.set("farSharpPlaneZ",   camera->depthOfFieldSettings().farSharpPlaneZ());
            args.set("farBlurryPlaneZ",  camera->depthOfFieldSettings().farBlurryPlaneZ());

            const float maxRadiusFraction = 
                max(max(camera->depthOfFieldSettings().nearBlurRadiusFraction(), camera->depthOfFieldSettings().farBlurRadiusFraction()), 0.001f);

            // This is a positive number
            const float nearNormalize =             
                (1.0f / (camera->depthOfFieldSettings().nearBlurryPlaneZ() - camera->depthOfFieldSettings().nearSharpPlaneZ())) *
                (camera->depthOfFieldSettings().nearBlurRadiusFraction() / maxRadiusFraction);
            alwaysAssertM(nearNormalize >= 0.0f, "Near normalization must be a non-negative factor");
            args.set("nearNormalize", nearNormalize); 

            // This is a positive number
            const float farNormalize =             
                (1.0f / (camera->depthOfFieldSettings().farSharpPlaneZ() - camera->depthOfFieldSettings().farBlurryPlaneZ())) *
                (camera->depthOfFieldSettings().farBlurRadiusFraction() / maxRadiusFraction);
            alwaysAssertM(farNormalize >= 0.0f, "Far normalization must be a non-negative factor");
            args.set("farNormalize", farNormalize);

        } else {
            args.set("focusPlaneZ", camera->depthOfFieldSettings().focusPlaneZ());
            args.set("scale", camera->imagePlanePixelsPerMeter(rd->viewport()) * 
                camera->depthOfFieldSettings().lensRadius() / (camera->depthOfFieldSettings().focusPlaneZ() * maxPhysicalBlurRadius(color->rect2DBounds())));

        }

        rd->applyRect(shader);

    } rd->pop2D();
}


void VVDoF::blurPass
(RenderDevice*                  rd,
 const shared_ptr<Texture>&     blurInput,
 const shared_ptr<Texture>&     nearInput,
 const shared_ptr<Framebuffer>& output,
 const shared_ptr<Shader>&      shader,
 const shared_ptr<Camera>&      camera) {

    alwaysAssertM(notNull(blurInput), "Input is NULL");

    // Dimension along which the blur fraction is measured
    const float dimension = 
        (camera->fieldOfViewDirection() == FOVDirection::HORIZONTAL) ?
        m_packedBuffer->width() : m_packedBuffer->height();

    const float maxRadiusFraction = 
            max(max(camera->depthOfFieldSettings().nearBlurRadiusFraction(), camera->depthOfFieldSettings().farBlurRadiusFraction()), 0.001f);

    const int maxCoCRadiusPixels =
            iCeil((camera->depthOfFieldSettings().model() == DepthOfFieldModel::ARTIST) ? 
                  (maxRadiusFraction * dimension) :
                  maxPhysicalBlurRadius(m_packedBuffer->rect2DBounds()));

    const int nearBlurRadiusPixels =
            iCeil((camera->depthOfFieldSettings().model() == DepthOfFieldModel::ARTIST) ? 
                  (camera->depthOfFieldSettings().nearBlurRadiusFraction() * dimension) :
                   maxPhysicalBlurRadius(m_packedBuffer->rect2DBounds()));

    rd->push2D(output); {
        rd->clear();
        Shader::ArgList& args = shader->args;

        args.set("blurSourceBuffer",   blurInput);
        args.set("nearSourceBuffer",   nearInput, true);
        args.set("maxCoCRadiusPixels", maxCoCRadiusPixels);
        args.set("nearBlurRadiusPixels", nearBlurRadiusPixels);
        args.set("invNearBlurRadiusPixels", 1.0f / nearBlurRadiusPixels);

        rd->applyRect(shader);
    } rd->pop2D();
}


void VVDoF::composite
(RenderDevice*   rd, 
 shared_ptr<Texture>    packedBuffer, 
 shared_ptr<Texture>    blurBuffer,
 shared_ptr<Texture>    nearBuffer,
 DebugOption     debugOption) {
    rd->push2D(); {
        rd->clear();
        Shader::ArgList& args = m_compositeShader->args;
        
        args.set("blurBuffer",   blurBuffer);
        args.set("nearBuffer",   nearBuffer);
        args.set("packedBuffer", packedBuffer);
        args.set("debugOption",  debugOption);
        
        rd->applyRect(m_compositeShader);
    } rd->pop2D();
}


/** Allocates or resizes a texture and framebuffer to match a target
    format and dimensions. */
static void matchTarget
(const shared_ptr<Texture>& target, 
 int                 divWidth, 
 int                 divHeight,
 const ImageFormat*  format,
 shared_ptr<Texture>&       texture, 
 shared_ptr<Framebuffer>&   framebuffer,
 Framebuffer::AttachmentPoint attachmentPoint,
 Texture::Settings   settings = Texture::Settings::buffer()) {
    alwaysAssertM(format, "Format may not be NULL");

    if (isNull(texture) || (texture->format() != format)) {
        // Allocate
        texture = Texture::createEmpty
            ("", 
             target->width() / divWidth, 
             target->height() / divHeight,
             format,
             Texture::DIM_2D_NPOT,
             settings);

        if (isNull(framebuffer)) {
            framebuffer = Framebuffer::create("");
        }
        framebuffer->set(attachmentPoint, texture);

    } else if ((texture->width() != target->width() / divWidth) ||
               (texture->height() != target->height() / divHeight)) {
        // Resize
        texture->resize(target->width(), target->height());
    }
}


void VVDoF::resizeBuffers(shared_ptr<Texture> target) {
    const ImageFormat* plusAlphaFormat = ImageFormat::getFormatWithAlpha(target->format());

    // Need an alpha channel for storing radius in the packed and far temp buffers
    matchTarget(target, 1, 1, plusAlphaFormat,     m_packedBuffer,    m_packedFramebuffer,     Framebuffer::COLOR0);

    matchTarget(target, 2, 1, plusAlphaFormat,     m_tempNearBuffer,  m_horizontalFramebuffer, Framebuffer::COLOR0);
    matchTarget(target, 2, 1, plusAlphaFormat,     m_tempBlurBuffer,  m_horizontalFramebuffer, Framebuffer::COLOR1);

    // Need an alpha channel (for coverage) in the near buffer
    matchTarget(target, 2, 2, plusAlphaFormat,     m_nearBuffer,      m_verticalFramebuffer,   Framebuffer::COLOR0, Texture::Settings::video());
    matchTarget(target, 2, 2, target->format(),    m_blurBuffer,      m_verticalFramebuffer,   Framebuffer::COLOR1, Texture::Settings::video());
}

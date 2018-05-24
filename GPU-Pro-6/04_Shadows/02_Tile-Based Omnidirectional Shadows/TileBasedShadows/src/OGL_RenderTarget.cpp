#include <stdafx.h>
#include <Demo.h>
#include <OGL_StructuredBuffer.h>
#include <OGL_RenderTarget.h>

// enum of color-buffer indices
static const GLenum colorBuffers[MAX_NUM_COLOR_BUFFERS]= 
{
  GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
  GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
  GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5,
  GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7 
};

void OGL_RenderTarget::Release()
{
  SAFE_DELETE_ARRAY(frameBufferTextures);
  SAFE_DELETE(depthStencilTexture);
  if(frameBufferName > 0)
    glDeleteFramebuffers(1, &frameBufferName); 
}

bool OGL_RenderTarget::Create(const RenderTargetDesc &desc)
{
  width = desc.width;
  height = desc.height;
  depth = desc.depth;	
  numColorBuffers = desc.CalcNumColorBuffers();
  const bool depthStencil = (desc.depthStencilBufferDesc.format != TEX_FORMAT_NONE);

  if((numColorBuffers > 0) || (depthStencil))
  {
    glGenFramebuffers(1, &frameBufferName);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferName);
  }

  if(numColorBuffers > 0)
  {
    clearMask |= COLOR_CLEAR_BIT;
    frameBufferTextures = new OGL_Texture[numColorBuffers];
    if(!frameBufferTextures)
      return false;
    for(unsigned int i=0; i<numColorBuffers; i++)
    {
      unsigned int flags = desc.colorBufferDescs[i].rtFlags;
      texFormats rtTexFormat = desc.colorBufferDescs[i].format;

      // On AMD graphics cards with Catalyst 15.7 WHQL driver SRGB to linear conversion seems to be broken. Therefore on AMD 
      // graphics cards the conversion is done manually in the shaders.
      if((!Demo::renderer->IsGpuAMD()) && (flags & SRGB_READ_RTF))
        rtTexFormat = Image::ConvertToSrgbFormat(desc.colorBufferDescs[i].format);
      if(rtTexFormat == TEX_FORMAT_NONE)
        return false;

      if(!frameBufferTextures[i].CreateRenderable(width, height, depth, rtTexFormat, desc.colorBufferDescs[i].rtFlags))
      {
        return false;  
      }
      glFramebufferTexture(GL_FRAMEBUFFER, colorBuffers[i], frameBufferTextures[i].textureName, 0);  		
    }
  }

  if(depthStencil)
  {
    if(desc.depthStencilBufferDesc.format == TEX_FORMAT_DEPTH24_STENCIL8)
      clearMask |= DEPTH_CLEAR_BIT | STENCIL_CLEAR_BIT;
    else
      clearMask |= DEPTH_CLEAR_BIT;
    depthStencilTexture = new OGL_Texture;
    if(!depthStencilTexture)
      return false;
    if(!depthStencilTexture->CreateRenderable(width, height, depth, desc.depthStencilBufferDesc.format, desc.depthStencilBufferDesc.rtFlags))
    {
      return false;
    }
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthStencilTexture->textureName, 0);  
    if(desc.depthStencilBufferDesc.format == TEX_FORMAT_DEPTH24_STENCIL8)
      glFramebufferTexture(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, depthStencilTexture->textureName, 0);  
  }

  if((numColorBuffers > 0) || (depthStencil))
  {
    GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false; 
  }
  
  return true;
}

bool OGL_RenderTarget::CreateBackBuffer()
{
  width = SCREEN_WIDTH;
  height = SCREEN_HEIGHT;  
  depth = 1;
  clearMask = COLOR_CLEAR_BIT | DEPTH_CLEAR_BIT | STENCIL_CLEAR_BIT;
  numColorBuffers = 1;	
  isBackBuffer = true;

  return true;
}

void OGL_RenderTarget::Bind(const RenderTargetConfig *rtConfig)
{
  if((isBackBuffer) || (frameBufferName > 0))
  {
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferName);
    glViewport(0, 0, width, height);
  }

  if(clearTarget)
  {
    Clear(clearMask);
    clearTarget = false;
  }

  bool srgb = false;
  if(!rtConfig)
  {
    if(frameBufferName > 0)
      glDrawBuffers(numColorBuffers, colorBuffers);
  }
  else
  {
    const RtConfigDesc &rtConfigDesc = rtConfig->GetDesc();
    if(isBackBuffer)
    {
      srgb = rtConfigDesc.flags & SRGB_WRITE_RTCF;
    }
    else
    {
      if(!(rtConfigDesc.flags & COMPUTE_RTCF))
      {
        srgb = rtConfigDesc.flags & SRGB_WRITE_RTCF;
        glDrawBuffers(rtConfigDesc.numColorBuffers, &colorBuffers[rtConfigDesc.firstColorBufferIndex]);
        assert(rtConfigDesc.numStructuredBuffers <= NUM_STRUCTURED_BUFFER_BP);
        for(unsigned int i=0; i<rtConfigDesc.numStructuredBuffers; i++)
          ((OGL_StructuredBuffer*)rtConfigDesc.structuredBuffers[i])->BindToRenderTarget(NUM_STRUCTURED_BUFFER_BP+i);
      }
      else
      {
        if(rtConfigDesc.numStructuredBuffers == 0)
        {
          assert(rtConfigDesc.numColorBuffers <= MAX_NUM_COLOR_BUFFERS);
          for(unsigned int i=0; i<rtConfigDesc.numColorBuffers; i++)
          {
            glBindImageTexture(i+NUM_STRUCTURED_BUFFER_BP, frameBufferTextures[i].textureName, 0, GL_TRUE, 0, GL_WRITE_ONLY,
              OGL_Texture::GetOglTexFormat(frameBufferTextures[i].format).internalFormat);
          }
        }
        else
        {
          assert(rtConfigDesc.numStructuredBuffers <= NUM_STRUCTURED_BUFFER_BP);
          for(unsigned int i=0; i<rtConfigDesc.numStructuredBuffers; i++)
            ((OGL_StructuredBuffer*)rtConfigDesc.structuredBuffers[i])->BindToRenderTarget(NUM_STRUCTURED_BUFFER_BP+i);
        }	
      }
    }  
  }  

  if(srgb)
    glEnable(GL_FRAMEBUFFER_SRGB);
  else
    glDisable(GL_FRAMEBUFFER_SRGB);
}

void OGL_RenderTarget::Clear(unsigned int newClearMask) const
{
  if((isBackBuffer) || (frameBufferName > 0))
    glClear(newClearMask);
}







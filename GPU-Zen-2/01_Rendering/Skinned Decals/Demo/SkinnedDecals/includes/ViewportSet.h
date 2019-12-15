#ifndef VIEWPORT_SET_H
#define VIEWPORT_SET_H

#include <render_states.h>

struct Viewport
{
  Viewport():
    topLeftX(0.0f),
    topLeftY(0.0f),
    width(0.0f),
    height(0.0f),
    minDepth(0.0f),
    maxDepth(1.0f)
  {
  }

  bool operator== (const Viewport &viewport) const
  {
    return ((topLeftX == viewport.topLeftX) && (topLeftY == viewport.topLeftY) && 
            (width == viewport.width) && (height == viewport.height) &&
            (minDepth == viewport.minDepth) && (maxDepth == viewport.maxDepth));
  }

  bool operator!= (const Viewport &viewport) const
  {
    return !((*this) == viewport);
  }

  float topLeftX, topLeftY, width, height, minDepth, maxDepth;
};

class ViewportSet
{
public:
  ViewportSet():
    numViewports(0)
  {
  }

  void Create(Viewport *viewports, UINT numViewports)
  {
    memcpy(this->viewports, viewports, sizeof(Viewport) * numViewports);
    this->numViewports = numViewports;
  }

  const Viewport* GetViewports() const
  {
    return viewports;
  }

  UINT GetNumViewports() const
  {
    return numViewports;
  }
 
private:
  Viewport viewports[MAX_NUM_VIEWPORTS];
  UINT numViewports;

};

#endif

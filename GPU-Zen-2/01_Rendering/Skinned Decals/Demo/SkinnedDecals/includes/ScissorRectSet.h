#ifndef SCISSOR_RECT_SET_H
#define SCISSOR_RECT_SET_H

#include <render_states.h>

struct ScissorRect
{
  ScissorRect():
    left(0),
    top(0),
    right(0),
    bottom(0)
  {
  }

  bool operator== (const ScissorRect &scissorRect) const
  {
    return ((left == scissorRect.left) && (top == scissorRect.top) && (right == scissorRect.right) && (bottom == scissorRect.bottom));
  }

  bool operator!= (const ScissorRect &scissorRect) const
  {
    return !((*this) == scissorRect);
  }

  LONG left, top, right, bottom;
};

class ScissorRectSet
{
public:
  ScissorRectSet():
    numScissorRects(0)
  {
  }

  void Create(ScissorRect *scissorRects, UINT numScissorRects)
  {
    memcpy(this->scissorRects, scissorRects, sizeof(ScissorRect) * numScissorRects);
    this->numScissorRects = numScissorRects;
  }

  const ScissorRect* GetScissorRects() const
  {
    return scissorRects;
  }

  UINT GetNumScissorRects() const
  {
    return numScissorRects;
  }
 
private:
  ScissorRect scissorRects[MAX_NUM_SCISSOR_RECTS];
  UINT numScissorRects;

};

#endif

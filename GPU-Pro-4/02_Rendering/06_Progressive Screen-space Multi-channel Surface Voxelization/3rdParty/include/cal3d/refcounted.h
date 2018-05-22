#ifndef CAL_REF_COUNTED_H
#define CAL_REF_COUNTED_H


#include "cal3d/platform.h"


namespace cal3d
{
    
  template<typename T> class RefPtr;
    
  /**
   * Derive from RefCounted to make your class have reference-counted
   * lifetime semantics.  Use RefPtr to manage references.  (Don't
   * call incRef() or decRef() directly.)  When deriving from
   * RefCounted, make your destructor protected so manual deletion
   * won't happen on accident.
   *
   * Note:  The reference count is initialized to 0.  This makes sense,
   * because, at object construction, no RefPtrs have referenced the
   * object.  However, this can cause trouble if you (indirectly) make
   * a RefPtr to 'this' within your constructor.  When the refptr goes
   * out of scope, the count goes back to 0, and the object is deleted
   * before it even exits the constructor.  Current recommended solution:  
   * Don't make refptrs to 'this'.  Pass 'this' by raw pointer and such.
   */
  class CAL3D_API RefCounted
  {
    template<typename T> friend T* explicitIncRef(T* p);
    friend void explicitDecRef(RefCounted* p);

  protected:
    RefCounted()
      : m_refCount(0)
    {
    }

    /**
     * Protected so users of refcounted classes don't use std::auto_ptr
     * or the delete operator.
     *
     * Interfaces that derive from RefCounted should define an inline,
     * empty, protected destructor as well.
     */
    virtual ~RefCounted()
    {
      assert(m_refCount == 0 && "_refCount nonzero in destructor");
    }

  // Must use RefPtr instead of manually calling incRef() and decRef().
  private:
    void incRef()
    {
      assert(m_refCount >= 0 && "_refCount is less than zero in incRef()!");
      ++m_refCount;
    }

    /**
     * Remove a reference from the internal reference count.  When this
     * reaches 0, the object is destroyed.
     */
    void decRef()
    {
      assert(m_refCount > 0 &&
             "_refCount is less than or equal to zero in decRef()!");
      if (--m_refCount == 0)
      {
        delete this;
      }
    }

  public:        
    int getRefCount() const
    {
      return m_refCount;
    }

  private:
    // Copying a RefCounted object must be done manually by the
    // subclass.  Otherwise the refCount gets copied too, and
    // that's Bad.
    RefCounted(const RefCounted& rhs);
    RefCounted& operator=(const RefCounted& rhs);
        
  private:
    int m_refCount;
  };

  template<typename T>
  T* explicitIncRef(T* p)
  {
    p->incRef();
    return p;
  }

  inline void explicitDecRef(RefCounted* p)
  {
    p->decRef();
  }

}


#endif

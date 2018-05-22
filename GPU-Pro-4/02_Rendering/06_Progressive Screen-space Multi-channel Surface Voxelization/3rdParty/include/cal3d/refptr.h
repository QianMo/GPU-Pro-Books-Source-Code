#ifndef CAL_REF_PTR_H
#define CAL_REF_PTR_H


namespace cal3d
{

    /// A container-safe smart pointer used for refcounted classes.
    template<typename T>
    class RefPtr
    {
    public:
        // For compatibility with Boost.Python.
        typedef T element_type;

        RefPtr(T* ptr = 0)
        {
            m_ptr = 0;
            *this = ptr;
        }

        RefPtr(const RefPtr<T>& ptr)
        {
            m_ptr = 0;
            *this = ptr;
        }

        ~RefPtr()
        {
            if (m_ptr)
            {
                explicitDecRef(m_ptr);
                m_ptr = 0;
            }
        }

        template<typename U>
        RefPtr<T>& operator=(U* ptr)
        {
            if (ptr != m_ptr)
            {
                if (m_ptr)
                {
                    explicitDecRef(m_ptr);
                }
                m_ptr = ptr;
                if (m_ptr)
                {
                    explicitIncRef(m_ptr);
                }
            }
            return *this;
        }

        template<typename U>
        RefPtr<T>& operator=(const RefPtr<U>& ptr)
        {
            *this = ptr.get();
            return *this;
        }

        /// Need this to override the built-in operator=
        RefPtr<T>& operator=(const RefPtr<T>& ptr)
        {
            *this = ptr.get();
            return *this;
        }

        /// Need this to override the built-in operator!
        bool operator!() const
        {            
            return !get();
        }

        T* operator->() const
        {
            assert(get() && "Accessing member of null pointer!");
            return get();
        }

        T& operator*() const
        {
            assert(get() && "Dereferencing null pointer!");
            return *get();
        }

        typedef RefPtr<T> this_type;

        /// Inspired by boost's smart_ptr facilities.
        typedef T* this_type::*unspecified_bool_type;

        /// This lets us write code like: if (ptr && ptr->valid())
        operator unspecified_bool_type() const
        {
            return (get() ? &this_type::m_ptr : 0);
        }

        T* get() const
        {
            assert(!m_ptr || m_ptr->getRefCount() > 0 &&
                   "Dereferencing pointer with refCount <= 0");
            return m_ptr;
        }

    private:
        T* m_ptr;
    };
    
    
    // For compatibility with Boost.Python.
    template<class T>
    T* get_pointer(const RefPtr<T>& p)
    {
        return p.get();
    }


    template<typename T, typename U>
    bool operator==(const RefPtr<T>& a, const RefPtr<U>& b)
    {
        return (a.get() == b.get());
    }

    template<typename T, typename U>
    bool operator==(const RefPtr<T>& a, const U* b)
    {
        return (a.get() == b);
    }

    template<typename T, typename U>
    bool operator==(const T* a, const RefPtr<U>& b)
    {
        return (a == b.get());
    }


    template<typename T, typename U>
    bool operator!=(const RefPtr<T>& a, const RefPtr<U>& b)
    {
        return (a.get() != b.get());
    }

    template<typename T, typename U>
    bool operator!=(const RefPtr<T>& a, const U* b)
    {
        return (a.get() != b);
    }

    template<typename T, typename U>
    bool operator!=(const T* a, const RefPtr<U>& b)
    {
        return (a != b.get());
    }
    
    
    template<typename T, typename U>
    bool operator<(const RefPtr<T>& a, const RefPtr<U>& b)
    {
        return (a.get() < b.get());
    }

    template<typename T, typename U>
    bool operator<(const RefPtr<T>& a, const U* b)
    {
        return (a.get() < b);
    }

    template<typename T, typename U>
    bool operator<(const T* a, const RefPtr<U>& b)
    {
        return (a < b.get());
    }
    
    
}


#endif

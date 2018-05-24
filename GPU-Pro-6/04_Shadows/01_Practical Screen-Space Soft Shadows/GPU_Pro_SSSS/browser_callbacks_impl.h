#ifndef browser_callbacks_impl_h
#define browser_callbacks_impl_h

#include <string.h>

#include "berkelium/WindowDelegate.hpp"

static std::wstring get_str( Berkelium::Script::Variant* args )
{
  if( args )
  {
    std::wstring str;

    if( args[0].toString().length() > 0 )
    {
      str.resize( args[0].toString().length() );
      memcpy( ( wchar_t* )str.data(), args[0].toString().data(), (args[0].toString().length()+1) * sizeof( wchar_t ) );
      str[args[0].toString().length()] = '\0';
    }

    return str;
  }
  else
  {
    return L"";
  }
}

template<class c, class t>
class operator_impl
{
  public:
    static void action( c f, Berkelium::Script::Variant* args );
};

template<class c>
class operator_impl<c, float>
{
  public:
    static void action( c f, Berkelium::Script::Variant* args )
    {
      f( args[0].toDouble() );
    }
};

template<class c>
class operator_impl<c, int>
{
  public:
    static void action( c f, Berkelium::Script::Variant* args )
    {
      f( args[0].toInteger() );
    }
};

template<class c>
class operator_impl<c, bool>
{
  public:
    static void action( c f, Berkelium::Script::Variant* args )
    {
      f( args[0].toBoolean() );
    }
};

template<class c>
class operator_impl<c, std::wstring>
{
  public:
    static void action( c f, Berkelium::Script::Variant* args )
    {
      if( args )
        f( get_str( args ) );
    }
};

template<class c>
class operator_impl<c, void*>
{
  public:
    static void action( c f, Berkelium::Script::Variant* args )
    {
      f( args );
    }
};

class functor_base
{
  public:
    virtual void operator()( Berkelium::Script::Variant* args ) {}
    virtual functor_base* clone()
    {
      return 0;
    }
    virtual ~functor_base() {}
};

template< class t >
class functor : public functor_base
{
    t func;

  public:
    void operator()( Berkelium::Script::Variant* args )
    {
      func();
    }

    functor_base* clone()
    {
      return new functor( *this );
    }

    functor( t f ) : func( f ) {}
};

template< class t, class u >
class functor_arg : public functor_base
{
    t func;

  public:
    void operator()( Berkelium::Script::Variant* args )
    {
      if( args )
        operator_impl<t, u>::action( func, args );
    }

    functor_base* clone()
    {
      return new functor_arg( *this );
    }

    functor_arg( t f, u d ) : func( f ) {}
};

template< class t >
class set_var
{
    t& var;
  public:
    void operator()( const t& v )
    {
      var = v;
    }

    set_var( t& v ) : var( v ) {}
};

class fun
{
    functor_base* ptr;

  public:
    void operator()( Berkelium::Script::Variant* args )
    {
      if( ptr )
        ( *ptr )( args );
    }

    template< class t >
    fun( t f ) : ptr( new functor< t >( f ) ) {}

    template< class t, class u >
    fun( t f, u d ) : ptr( new functor_arg< t, u >( f, d ) ) {}

    //default constructor
    fun() : ptr( 0 ) {}

    //assignment operator
    fun& operator=( fun f )
    {
      std::swap( ptr, f.ptr );
      return *this;
    }

    //copy constructor
    fun( const fun& f )
    {
      ptr = f.ptr->clone();
    }

    //move constructor
    fun( fun && f )
    {
      ptr = f.ptr;
      f.ptr = 0;
    }

    //destructor
    ~fun()
    {
      delete ptr;
    }
};

#endif

/*****************************************************/
/* lean Macros                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_DEFAULT
#define LEAN_DEFAULT

/// @addtogroup GlobalMacros
/// @{

#define LEAN_NOP_COPY(type) type(const type&) noexcept { }
#define LEAN_NOP_COPY_DECL(modifiers, type) modifiers type(const type&) noexcept;
#define LEAN_NOP_COPY_DEF(type) type::type(const type&) noexcept { }

#define LEAN_NOP_ASSIGN(type) type& operator =(const type&) noexcept { return *this; }
#define LEAN_NOP_ASSIGN_DECL(modifiers, type) modifiers type& operator =(const type&) noexcept;
#define LEAN_NOP_ASSIGN_DEF(type) type& type::operator =(const type&) noexcept { return *this; }

#define LEAN_DEFAULT_DTOR(type) ~type() noexcept { }
#define LEAN_DEFAULT_DTOR_DECL(modifiers, type) modifiers ~type() noexcept;
#define LEAN_DEFAULT_DTOR_DEF(type) type::~type() noexcept { }

#ifndef LEAN_OPTIMIZE_DEFAULT_DESTRUCTOR
	#define LEAN_OPT_DEFAULT_DTOR(type) LEAN_DEFAULT_DTOR(type)
#else
	#define LEAN_OPT_DEFAULT_DTOR(type)
#endif

#define LEAN_NOP_COPY_ASSIGN(type) \
	LEAN_NOP_COPY(type) \
	LEAN_NOP_ASSIGN(type)

#define LEAN_NOP_ASSIGN_DEFAULT_DTOR(type) \
	LEAN_NOP_ASSIGN(type) \
	LEAN_DEFAULT_DTOR(type)

#define LEAN_NOP_COPY_ASSIGN_DEFAULT_DTOR(type) \
	LEAN_NOP_COPY(type) \
	LEAN_NOP_ASSIGN(type) \
	LEAN_DEFAULT_DTOR(type)

#define LEAN_NOP_ASSIGN_OPT_DEFAULT_DTOR(type) \
	LEAN_NOP_ASSIGN(type) \
	LEAN_OPT_DEFAULT_DTOR(type)

#define LEAN_NOP_COPY_ASSIGN_OPT_DEFAULT_DTOR(type) \
	LEAN_NOP_COPY(type) \
	LEAN_NOP_ASSIGN(type) \
	LEAN_OPT_DEFAULT_DTOR(type)

/// @}

#endif
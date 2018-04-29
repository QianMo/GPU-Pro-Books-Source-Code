#ifndef __MACRO_EXCEPTION_HPP__
#define __MACRO_EXCEPTION_HPP__

#include <exception>

#define DEFINECPPEXCEPTION(ClassName) \
class ClassName : public std::exception { \
public : ClassName(const char* pText):m_pText(pText){}\
	const char* m_pText;\
	virtual const char* what() const throw (){ return m_pText; } \
	~ClassName(){} };


#endif
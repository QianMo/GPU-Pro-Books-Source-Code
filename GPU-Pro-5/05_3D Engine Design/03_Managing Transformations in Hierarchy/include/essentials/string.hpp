#ifndef ESSENTIALS_STRING_HPP
#define ESSENTIALS_STRING_HPP

#include <sstream>



template <class TYPE>
string ToString(const TYPE& arg)
{
	ostringstream out;
	out << arg;
	return out.str();
}



#endif

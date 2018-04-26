#include <sstream>

//namespace std{
namespace std
{
	template <class T, class chartype>
	inline basic_string<chartype> toTString(const T& t) {
		basic_ostringstream<chartype> o;
		o << t;
		return o.str();
	}
	template <class T>
	inline string toString(const T& t) {
		return toTString<T,char>(t);
	}
	template <class T>
	inline wstring toWString(const T& t) {
		return toTString<T,wchar_t>(t);
	}  
}
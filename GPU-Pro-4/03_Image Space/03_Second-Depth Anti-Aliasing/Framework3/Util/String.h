
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef __STRING_H__
#define __STRING_H__

// To make MSVC 2005 happy
#define _CRT_SECURE_NO_DEPRECATE
#include <string.h>
#include <stdlib.h>

class String {
public:
	String();
	String(unsigned int cap);
	String(const char *string);
	String(const char *string, unsigned int len);
	String(const String &string);
	~String();
	
	void setLength(const unsigned int newLength);

	operator const char *() const { return str; }
	unsigned int getLength() const { return length; }
	bool isEmpty() const { return length == 0; }

	int toInt() const { return atoi(str); }
	float toFloat() const { return (float) atof(str); }
	double toDouble() const { return atof(str); }

	void makeLowerCase();

	void operator =  (const char *string);
	void operator =  (const String &string);
	void operator += (const char *string);
	void operator += (const String &string);

	void assign(const char *string, const unsigned int len);
	void append(const char *string, const unsigned int len);
	void appendInt(const int integer);
	bool insert(const unsigned int pos, const char *string, const unsigned int len);
	bool remove(const unsigned int pos, const unsigned int len);

	unsigned int replace(const char oldCh, const char newCh);
	unsigned int replace(const char *oldStr, const char *newStr);

	bool  find(const char ch, unsigned int pos = 0, unsigned int *index = NULL) const;
	bool rfind(const char ch, int pos = -1, unsigned int *index = NULL) const;
	bool  find(const char *string, unsigned int pos = 0, unsigned int *index = NULL) const;

	void trimRight(const char *chars);

	void sprintf(char *formatStr, ... );
private:
	char *str;
	unsigned int length;
	unsigned int capasity;
};

String operator + (const String &string, const String &string2);
String operator + (const String &string, const char   *string2);
String operator + (const char   *string, const String &string2);

bool operator == (const String &string, const String &string2);
bool operator == (const String &string, const char   *string2);
bool operator == (const char   *string, const String &string2);

bool operator != (const String &string, const String &string2);
bool operator != (const String &string, const char   *string2);
bool operator != (const char   *string, const String &string2);

bool operator > (const String &string, const String &string2);
bool operator > (const String &string, const char   *string2);
bool operator > (const char   *string, const String &string2);

bool operator < (const String &string, const String &string2);
bool operator < (const String &string, const char   *string2);
bool operator < (const char   *string, const String &string2);

bool operator >= (const String &string, const String &string2);
bool operator >= (const String &string, const char   *string2);
bool operator >= (const char   *string, const String &string2);

bool operator <= (const String &string, const String &string2);
bool operator <= (const String &string, const char   *string2);
bool operator <= (const char   *string, const String &string2);

#endif // __STRING_H__

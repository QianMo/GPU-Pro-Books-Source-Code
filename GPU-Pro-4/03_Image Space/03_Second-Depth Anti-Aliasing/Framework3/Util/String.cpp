
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

#include "String.h"
#include <stdarg.h>
#include <stdio.h>

int getAppropriateCapasity(unsigned int capasity, unsigned int min){
	do {
		capasity <<= 1;
	} while (capasity < min);
	return capasity;
}


String::String(){
	length = 0;
	str = (char *) malloc(capasity = 32);
	str[0] = '\0';
}

String::String(unsigned int cap){
	length = 0;
	str = (char *) malloc(capasity = cap);
	str[0] = '\0';
}

String::String(const char *string){
	length = (unsigned int) strlen(string);
	str = (char *) malloc(capasity = length + 1);
	strncpy(str, string, capasity);
}

String::String(const char *string, unsigned int len){
	length = len;
	str = (char *) malloc(capasity = len + 1);
	strncpy(str, string, len);
	str[len] = '\0';
}

String::String(const String &string){
	length = string.length;
	capasity = string.capasity;
	str = (char *) malloc(capasity);
	strncpy(str, string.str, length + 1);
}


String::~String(){
	free(str);
}

void String::setLength(const unsigned int newLength){
	str = (char *) realloc(str, capasity = newLength + 1);
	str[length = newLength] = '\0';
}


void String::makeLowerCase(){
	for (unsigned int i = 0; i < length; i++){
		if (str[i] >= 'A' && str[i] <= 'Z') str[i] -= ('A' - 'a');
	}
}


void String::operator = (const char *string){
	assign(string, (unsigned int) strlen(string));
}

void String::operator = (const String &string){
	assign(string.str, string.length);
}

void String::operator += (const char *string){
	append(string, (unsigned int) strlen(string));
}

void String::operator += (const String &string){
	append(string.str, string.length);
}

void String::assign(const char *string, const unsigned int len){
	if (len + 1 >= capasity){
		capasity = getAppropriateCapasity(capasity, len + 1);
		free(str);
		str = (char *) malloc(capasity);
	}

	strncpy(str, string, len);
	str[len] = '\0';
	length = len;
}

void String::append(const char *string, const unsigned int len){
	unsigned int newLength = length + len;
	if (newLength + 1 >= capasity){
		capasity = getAppropriateCapasity(capasity, newLength + 1);
		str = (char *) realloc(str, capasity);
	}

	strncpy(str + length, string, len);
	str[newLength] = '\0';

	length = newLength;
}

void String::appendInt(const int integer){
	char s[16];
	append(s, ::sprintf(s, "%d", integer));
}

bool String::insert(const unsigned int pos, const char *string, const unsigned int len){
	if (pos > length) return false;

	unsigned int newLength = length + len;
	if (newLength + 1 >= capasity){
		capasity = getAppropriateCapasity(capasity, newLength + 1);
		str = (char *) realloc(str, capasity);
	}

	unsigned int n = length - pos;
	for (unsigned int i = 0; i <= n; i++){
		str[newLength - i] = str[length - i];
	}
	strncpy(str + pos, string, len);

	length = newLength;
	return true;
}

bool String::remove(const unsigned int pos, const unsigned int len){
	if (pos + len > length) return false;

	length -= len;
	unsigned int n = length - pos;
	for (unsigned int i = 0; i <= n; i++){
		str[pos + i] = str[pos + len + i];
	}

	return true;
}

unsigned int String::replace(const char oldCh, const char newCh){
	unsigned int count = 0;
	for (unsigned int i = 0; i < length; i++){
		if (str[i] == oldCh){
			str[i] = newCh;
			count++;			
		}
	}
	return count;
}

unsigned int String::replace(const char *oldStr, const char *newStr){
	unsigned int i, j, last, count = 0;
	const unsigned int oldLen = (unsigned int) strlen(oldStr);
	const unsigned int newLen = (unsigned int) strlen(newStr);

	for (i = 0; i < length - oldLen; i++){
		if (strncmp(str + i, oldStr, oldLen) == 0) count++;
	}

	length += (newLen - oldLen) * count;
	capasity = getAppropriateCapasity(capasity, length + 1);

	char *dest = (newLen > oldLen)? (char *) malloc(capasity) : str;
	char *newString = dest;

	i = 0;
	last = 0;
	for (j = 0; j < count; j++){
		while (strncmp(str + i, oldStr, oldLen) != 0) i++;

		if (j || newLen > oldLen) strncpy(dest, str + last, i - last);
		dest += i - last;
		strncpy(dest, newStr, newLen);
		dest += newLen;
		i += oldLen;
		last = i;
	}

	strcpy(dest, str + i);

	if (newLen > oldLen){
		free(str);
		str = newString;
	}

	return count;
}

bool String::find(const char ch, unsigned int pos, unsigned int *index) const {
	for (unsigned int i = pos; i < length; i++){
		if (str[i] == ch){
			if (index != NULL) *index = i;
			return true;
		}
	}
	return false;
}

bool String::rfind(const char ch, int pos, unsigned int *index) const {
	unsigned int i = (pos < 0)? length : pos;

	while (i){
		i--;
		if (str[i] == ch){
			if (index != NULL) *index = i;
			return true;
		}
	}
	return false;
}

bool String::find(const char *string, unsigned int pos, unsigned int *index) const {
	char *st = strstr(str + pos, string);
	if (st != NULL){
		if (index != NULL) *index = (unsigned int) (st - str);
		return true;
	}
	return false;
}


bool isInString(char *str, const char ch){
	while (*str){
		if (*str++ == ch) return true;
	}
	return false;
}

void String::trimRight(const char *chars){
	int i = length - 1;
	while (i >= 0 && isInString((char *) chars, str[i])) i--;

	length = i + 1;
	str[length] = '\0';
}

void String::sprintf(char *formatStr, ... ){
	char buf[256];
	union {
		double farg;
		int    iarg;
		char  *sarg;
	};
	va_list marker;
	va_start(marker, formatStr);

	int len, i = 0;
	int last = 0;
	int flen = (int) strlen(formatStr);
	do {
		while (i < flen && formatStr[i] != '%') i++;
		append(formatStr + last, i - last);
		if (i >= flen){
			va_end(marker);
			return;
		}

		switch (formatStr[i + 1]){
		case 'd':
			iarg = va_arg(marker, int);
			len = ::sprintf(buf, "%d", iarg);
			append(buf, len);
			break;
		case 'f':
			farg = va_arg(marker, double);
			len = ::sprintf(buf, "%f", farg);
			append(buf, len);
			break;
		case 'g':
			farg = va_arg(marker, double);
			len = ::sprintf(buf, "%g", farg);
			append(buf, len);
			break;
		case 's':
			sarg = va_arg(marker, char *);
			operator += (sarg);
			break;
		case 'x':
			iarg = va_arg(marker, int);
			len = ::sprintf(buf, "%x", iarg);
			append(buf, len);
			break;
		case 'X':
			iarg = va_arg(marker, int);
			len = ::sprintf(buf, "%X", iarg);
			append(buf, len);
			break;
		case '%':
			append("%", 1);
			break;
		}
		i += 2;
		last = i;
	} while (true);
}


String operator + (const String &string, const String &string2){
	String str(string.getLength() + string2.getLength() + 1);
	str = string;
	str.append(string2, string2.getLength());
	return str;
}

String operator + (const String &string, const char *string2){
	int len = (int) strlen(string2);
	String str(string.getLength() + len + 1);
	str = string;
	str.append(string2, len);
	return str;
}

String operator + (const char *string, const String &string2){
	int len = (int) strlen(string);
	String str(string2.getLength() + len + 1);
	str = string;
	str.append(string2, string2.getLength());
	return str;
}

// Quick implementation macro for all comparison functions
#define IMP_COMPARE(s0,op,s1) bool operator op (const s0 string, const s1 string2){ return (strcmp(string, string2) op 0); }

IMP_COMPARE(String &, ==, String &)
IMP_COMPARE(String &, ==, char *)
IMP_COMPARE(char *  , ==, String &)

IMP_COMPARE(String &, !=, String &)
IMP_COMPARE(String &, !=, char *)
IMP_COMPARE(char *  , !=, String &)

IMP_COMPARE(String &, >, String &)
IMP_COMPARE(String &, >, char *)
IMP_COMPARE(char *  , >, String &)

IMP_COMPARE(String &, <, String &)
IMP_COMPARE(String &, <, char *)
IMP_COMPARE(char *  , <, String &)

IMP_COMPARE(String &, >=, String &)
IMP_COMPARE(String &, >=, char *)
IMP_COMPARE(char *  , >=, String &)

IMP_COMPARE(String &, <=, String &)
IMP_COMPARE(String &, <=, char *)
IMP_COMPARE(char *  , <=, String &)

//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Lexer.h"
#include "Range.h"
#include "Errors.h"

namespace D3DEffectsLite
{

template <class Element>
struct checked_ptr
{
	typedef Element value_type;
	typedef Element* pointer;

	pointer it;
	pointer end;

	checked_ptr(pointer it, pointer end)
		: it(it), end(end) { }

	checked_ptr& operator +=(ptrdiff_t d) { it += d; return *this; }
	checked_ptr& operator -=(ptrdiff_t d) { it -= d; return *this; }

	checked_ptr& operator ++() { ++it; return *this; }
	checked_ptr& operator --() { --it; return *this; }

	checked_ptr operator ++(int) { checked_ptr p(*this); ++it; return p; }
	checked_ptr operator --(int) { checked_ptr p(*this); --it; return p; }

	const value_type operator *() const { return (it < end) ? *it : 0; }
	operator pointer() const { return it; }

	const value_type operator [](ptrdiff_t pos) const
	{
		return (pos < end - it) ? it[pos] : 0;
	}
};

/// True, if the given character marks a line break.
bool is_endl(char c)
{
	return c == '\n';
}
/// True, if the given character makes a valid beginning for an identifier.
inline bool is_identifier_begin(int c)
{
	return isalpha(c) || c == '_';
}
/// True, if the given character is part of a valid identifier.
inline bool is_identifier(int c)
{
	return isalnum(c) || c == '_';
}

/// True, if the given character makes a valid beginning for a numeric literal.
inline bool is_num_literal_begin(int c)
{
	return isdigit(c) != 0;
}
/// True, if the given character is part of a valid numeric literal.
inline bool is_num_literal(int c)
{
	return isalnum(c) || c == '.';
}

/// True, if the given character is part of a special character sequence.
inline bool is_special(int c)
{
	return ispunct(c) && c != '_';
}

/// Counts the number of backslashes starting at the given position, checking backwards.
template <class Iterator>
inline size_t count_backslashes(Iterator it)
{
	size_t count = 0;

	while (*it == '\\')
	{
		++count;
		--it;
	}

	return count;
}

/// Extracts the identifier stating at the given position.
Token extract_identifier(checked_ptr<const char> &nextChar)
{
	Token token;

	const char *identifierBegin = nextChar++;

	while (is_identifier(*nextChar))
		++nextChar;

	token.range = CharRange(identifierBegin, nextChar);
	token.type = TokenType::identifier;

	return token;
}

/// Extracts the numeric literal stating at the given position.
Token extract_num_literal(checked_ptr<const char> &nextChar)
{
	Token token;

	const char *numericBegin = nextChar++;

	while (is_num_literal(*nextChar))
		++nextChar;

	token.range = CharRange(numericBegin, nextChar);
	token.type = TokenType::number;

	return token;
}

/// Extracts the character literal stating at the given position.
Token extract_char_literal(checked_ptr<const char> &nextChar)
{
	Token token;

	const char *numericBegin = nextChar++;

	while (isprint(*nextChar) && (*nextChar != '\'' || count_backslashes(nextChar - 1) % 2 == 1))
		++nextChar;
	
	// Don't forget to extract terminating character
	if (*nextChar)
		++nextChar;

	token.range = CharRange(numericBegin, nextChar);
	token.type = TokenType::character;
	
	return token;
}

/// Extracts the character literal stating at the given position.
Token extract_string_literal(checked_ptr<const char> &nextChar)
{
	Token token;

	const char *numericBegin = nextChar++;

	while (isprint(*nextChar) && (*nextChar != '"' || count_backslashes(nextChar - 1) % 2 == 1))
		++nextChar;
	
	// Don't forget to extract terminating character
	if (*nextChar)
		++nextChar;

	token.range = CharRange(numericBegin, nextChar);
	token.type = TokenType::string;
	
	return token;
}

/// Extracts the special character sequence starting at the given position.
Token extract_special(checked_ptr<const char> &nextChar)
{
	Token token;

	const char *specialBegin = nextChar++;

	// The only compound operators we need, so far
	if (*specialBegin == '-')
	{
		if (*nextChar == '>' || *nextChar == '-')
			++nextChar;
	}
	else if (*specialBegin == '.' && *nextChar == '.' && nextChar[1] == '.')
		nextChar += 2;
	
	token.range = CharRange(specialBegin, nextChar);
	token.type = TokenType::punct;

	return token;
}

/// Skips the comment starting at the given position.
void skip_line_comment(checked_ptr<const char> &nextChar)
{
	while (*++nextChar && !is_endl(*nextChar));
}

/// Skips the comment starting at the given position.
void skip_multiline_comment(checked_ptr<const char> &nextChar)
{
	while (*++nextChar && (*nextChar != '/' || nextChar[-1] != '*'));

	// Don't forget to extract terminating character
	if (*nextChar)
		++nextChar;
}

/// Gets the next token.
Token next_token(LexerCursor &cursor, const char *end, unsigned int flags)
{
	Token token;

	checked_ptr<const char> nextChar(cursor.nextChar, end);

	bool skipEndl = (flags & ParserFlags::keep_endl) == 0;
	TokenType::T eof = (flags & ParserFlags::end_is_endl) == ParserFlags::end_is_endl ? TokenType::endline : TokenType::end;
	bool skipChars = true;

	// Continue skipping until neither white spaces nor comments
	while (skipChars)
	{
		// Skip whitespaces
		while (isspace(*nextChar) && (skipEndl || !is_endl(*nextChar)))
			++nextChar;

		skipChars = false;

		// Skip comments
		if (*nextChar == '/')
		{
			if (nextChar[1] == '/')
			{
				skip_line_comment(++nextChar);
				skipChars = true;
			}
			else if (nextChar[1] == '*')
			{
				skip_multiline_comment(++nextChar);
				skipChars = true;
			}
		}
	}

	if (*nextChar)
	{
		if (is_endl(*nextChar))
		{
			token.type = TokenType::endline;
			token.range = CharRange(nextChar, nextChar + 1);
			++nextChar;
		}
		// Pass on to specific token lexers
		else if (is_identifier_begin(*nextChar))
			token = extract_identifier(nextChar);
		else if (is_num_literal_begin(*nextChar))
			token = extract_num_literal(nextChar);
		else if (*nextChar == '\'')
			token = extract_char_literal(nextChar);
		else if (*nextChar == '"')
			token = extract_string_literal(nextChar);
		else
			token = extract_special(nextChar);
	}
	else
	{
		// End of file
		token.type = eof;
		token.range = CharRange(nextChar, nextChar);
	}

	assert(token.range.begin && token.range.end);

	// Update cursor w/ pointer & line number
	while (cursor.nextChar < nextChar)
		cursor.line += is_endl(*cursor.nextChar++);

	return token;
}

/// Gets the next token.
Token& next_token(LexerState &state, unsigned int flags)
{
	if (state.isAccepted || state.openFlags != flags)
	{
		// Re-read from first open character onwards when flags have changed
		if (!state.isAccepted)
			state.cursor = state.openCursor;
		else
			state.openCursor = state.cursor;

		state.openFlags = flags;
		state.token = next_token(state.cursor, state.endChar, flags);
		state.isAccepted = false;
	}

	return state.token;
}
/// Accepts the current token.
Token& accept_token(LexerState &state)
{
	state.isAccepted = true;
	state.accepted = state.token;
	state.acceptedCursor = state.cursor;
	return state.token;
}

} // namespace

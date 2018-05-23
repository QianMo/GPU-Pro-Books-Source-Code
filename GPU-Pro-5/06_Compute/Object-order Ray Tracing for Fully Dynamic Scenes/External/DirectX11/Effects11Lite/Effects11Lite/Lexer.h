//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Range.h"

namespace D3DEffectsLite
{

/// True, if the given character marks a line break.
bool is_endl(char c);

struct TokenType
{
	enum T
	{
		identifier,
		number,
		character,
		string,
		punct,
		endline,
		end
	};
};

struct Token
{
	TokenType::T type;
	CharRange range;

	Token() { }
	Token(TokenType::T type, CharRange range)
		: type(type), range(range) { }
};

inline bool operator ==(const Token &t1, const Token &t2)
{
	return t1.type == t2.type && t1.range == t2.range;
}
inline bool operator !=(const Token &t1, const Token &t2)
{
	return !(t1 == t2);
}

struct ParserFlags
{
	enum T
	{
		none = 0,
		keep_endl = 0x1,
		end_is_endl = 0x2 | keep_endl
	};
};

/// Gets the next token.
Token next_token(const char* &nextChar, const char *end, unsigned int flags = ParserFlags::none);

struct LexerCursor
{
	const char *nextChar;
	unsigned int line;

	LexerCursor(const char *nextChar, unsigned int line)
		: nextChar(nextChar),
		line(line) { }
};

struct LexerState
{
	const char *endChar;
	LexerCursor cursor;
	Token token;
	LexerCursor acceptedCursor;
	Token accepted;
	LexerCursor openCursor;
	unsigned int openFlags;
	bool isAccepted;

	LexerState(LexerCursor src, const char *srcEnd, unsigned int flags = ParserFlags::none)
		: endChar(srcEnd),
		cursor(src),
		acceptedCursor(src),
		isAccepted(true),
		openCursor(src),
		openFlags(flags) { }
};

/// Gets the next token.
Token& next_token(LexerState &state, unsigned int flags = ParserFlags::none);
/// Accepts the current token.
Token& accept_token(LexerState &state);
/// Reverts the current token.
Token& revert_token(LexerState &state);

struct AcceptOperator
{
	enum T { Identiy, Not };
};
inline bool apply(bool v, AcceptOperator::T op)
{
	return (op == AcceptOperator::Not) ? !v : v;
}

/// Checks for the given token type.
inline bool check_token(LexerState &state, TokenType::T type)
{
	return state.token.type == type;
}
/// Checks for the given token.
inline bool check_token(LexerState &state, const Token &token)
{
	return state.token == token;
}

/// Checks for the given token type.
inline bool check_next_token(LexerState &state, TokenType::T type, unsigned int flags = ParserFlags::none)
{
	return next_token(state, flags).type == type;
}
/// Checks for the given token.
inline bool check_next_token(LexerState &state, const Token &token, unsigned int flags = ParserFlags::none)
{
	return next_token(state, flags) == token;
}

/// Accepts the given token type.
inline bool accept_next_token(LexerState &state, TokenType::T type, unsigned int flags = ParserFlags::none, AcceptOperator::T op = AcceptOperator::Identiy)
{
	bool accepted = apply(check_next_token(state, type, flags), op);
	if (accepted)
		accept_token(state);
	return accepted;
}
/// Accepts the given token.
inline bool accept_next_token(LexerState &state, const Token &token, unsigned int flags = ParserFlags::none, AcceptOperator::T op = AcceptOperator::Identiy)
{
	bool accepted = apply(check_next_token(state, token, flags), op);
	if (accepted)
		accept_token(state);
	return accepted;
}

/// Accepts the given token type.
inline bool dont_accept_next_token(LexerState &state, TokenType::T type, unsigned int flags = ParserFlags::none)
{
	return accept_next_token(state, type, flags, AcceptOperator::Not);
}
/// Accepts the given token.
inline bool dont_accept_next_token(LexerState &state, const Token &token, unsigned int flags = ParserFlags::none)
{
	return accept_next_token(state, token, flags, AcceptOperator::Not);
}

} // namespace

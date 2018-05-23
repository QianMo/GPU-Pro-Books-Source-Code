//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "D3DEffectsLiteHooks.h"
#include "Lexer.h"
#include "Errors.h"
#include "Range.h"
#include "MemBlob.h"
#include <list>
#include <vector>

namespace D3DEffectsLite
{

namespace
{

/// Standard containers modified with custom allocator.
namespace stdm
{

typedef std::basic_string< char, std::char_traits<char>, custom_allocator<char> > string;
template <class T> struct vector_t { typedef std::vector< T, custom_allocator<T> > type; };
template <class T> struct list_t { typedef std::list< T, custom_allocator<T> > type; };

} // namespace

typedef stdm::list_t<stdm::string>::type strings_t;
typedef stdm::vector_t<CharRange>::type ranges_t;
typedef stdm::vector_t<AnnotedCharRange>::type annoted_ranges_t;

/// Moves the given value to the back of the given container, passing the given allocator.
template <class Container, class T, class A>
T& add_by_swap(Container &strings, T &string, const A &alloc)
{
	strings.push_back(T(alloc));
	strings.back().swap(string);
	return strings.back();
}

/// Moves the given value to the back of the given container.
template <class Container, class T>
inline T& add_by_swap(Container &strings, T &string)
{
	return add_by_swap(strings, string, string.get_allocator());
}

/// Adds the given number to the given string pool.
inline const stdm::string& add_num(strings_t &strings, int num, Allocator *alloc)
{
	char numString[32];
	return add_by_swap(strings, construct_ref(stdm::string)(numString, itoa(num, numString, 10), alloc));
}

/// Adds the given number to the given string pool.
inline const stdm::string& add_line(strings_t &strings, const char *srcName, int line, Allocator *alloc)
{
	char numString[4096];
	// TODO: Unsafe, MS has no snprintf ...
	sprintf(numString, "\n#line %d \"%s\"\n", line, srcName);
	return add_by_swap(strings, construct_ref(stdm::string)(numString, alloc));
}

/// Merges the given collection of character ranges to one string, using the given allocator.
template <class Container>
stdm::string merge_ranges(const Container &ranges, Allocator *alloc)
{
	stdm::string result(alloc);
	size_t count = 0;

	for (typename Container::const_iterator it = ranges.begin(), itEnd = ranges.end(); it != itEnd; ++it)
		count += it->end - it->begin;

	result.reserve(count);

	for (typename Container::const_iterator it = ranges.begin(), itEnd = ranges.end(); it != itEnd; ++it)
		result.append(it->begin, it->end);

	return result;
}

/// Merges the given collection of character ranges to one string, using the given allocator.
template <class Container, class AnnotedRanges>
stdm::string& merge_ranges(const Container &ranges, stdm::string &result, AnnotedRanges &annotations, Allocator *alloc)
{
	merge_ranges(ranges, alloc).swap(result);
	
	annotations.reserve(ranges.size());
	const char *resultCursor = &result[0];

	for (typename Container::const_iterator it = ranges.begin(), itEnd = ranges.end(); it != itEnd; ++it)
	{
		const char *nextResultCursor = resultCursor + size(*it);
		annotations.push_back( AnnotedCharRange( CharRange(resultCursor, nextResultCursor), it->annot ) );
		resultCursor = nextResultCursor;
	}

	return result;
}

const CharRangeAnnotation KeepAnnotations(nullptr, -1);

/// Merges the given collection of character ranges to one string, using the given allocator.
stdm::string fix_lines(const AnnotedCharRange *ranges, const AnnotedCharRange *rangesEnd, Allocator *alloc)
{
	strings_t stringPool(alloc);

	ranges_t processedRanges(alloc);
	processedRanges.reserve( 2 * (rangesEnd - ranges) );

	const AnnotedCharRange *prevRange = nullptr;
	int nextLine = 1;

	for (const AnnotedCharRange *it = ranges; it < rangesEnd; ++it)
	{
		if (it->annot.name && (!prevRange || prevRange->annot.name != it->annot.name || nextLine != it->annot.line))
		{
			processedRanges.push_back( CharRange( add_line(stringPool, it->annot.name, it->annot.line, alloc) ) );
			
			prevRange = it;
			nextLine = it->annot.line;
		}
		for (const char *j = it->begin; j < it->end; ++j)
			nextLine += is_endl(*j);
		processedRanges.push_back( *it );
	}

	return merge_ranges(processedRanges, alloc);
}

/// String with annotation ranges.
struct AnnotedString
{
	stdm::string text;
	annoted_ranges_t annotations;

	AnnotedString(Allocator *alloc)
		: text(alloc),
		annotations(alloc) { }

	// TODO: Should never be called, but rather be omitted by NRVO!
	AnnotedString(const AnnotedString &right)
		: text(right.text),
		annotations(right.annotations)
	{
		for (annoted_ranges_t::iterator it = annotations.begin(), itEnd = annotations.end(); it < itEnd; ++it)
		{
			it->begin = &text[0] + (it->begin - &right.text[0]);
			it->end = &text[0] + (it->end - &right.text[0]);
		}
	}

	AnnotedString& operator =(AnnotedString right)
	{
		text.swap(right.text);
		annotations.swap(right.annotations);
	}
};

/// Merges the given collection of character ranges to one string, using the given allocator.
template <class Container>
AnnotedString merge_annoted_ranges(const Container &ranges, Allocator *alloc)
{
	AnnotedString result(alloc);
	merge_ranges(ranges, result.text, result.annotations, alloc);
	return result;
}

struct OpenIncludeError : LoggedError { };

struct IncludeFile
{
	Include *include;
	const char *src;
	UINT srcLength;

	IncludeFile(Include *include, IncludeType::T type, const char *fileName, const void *parent)
		: include(include)
	{
		assert(include);

		if (FAILED(include->Open(type, fileName, parent, reinterpret_cast<const void**>(&src), &srcLength)))
		{
			LogLine("cannot open hook include ", fileName);
			throw OpenIncludeError();
		}
	}
	~IncludeFile()
	{
		assert(include);
		include->Close(src);
	}

private:
	IncludeFile(const IncludeFile&);
	IncludeFile& operator =(const IncludeFile&);
};

struct HookDef
{
	CharRange name;
	AnnotedCharRange defs;
};

struct HookFun
{
	CharRange name;
	CharRange decl;
	CharRange stateType;
	CharRange call;
};

struct HookState
{
	stdm::string hookID;

	CharRange configuration;

	typedef stdm::vector_t<HookDef>::type definitions_t;
	definitions_t definitions;

	typedef stdm::vector_t<HookFun>::type functions_t;
	functions_t functions;

	HookState(Allocator *alloc)
		: hookID(alloc),
		definitions(alloc),
		functions(alloc) { }
};

typedef stdm::list_t<HookState>::type hookstate_t;

inline stdm::string MakeHookID(UINT counter, Allocator *alloc)
{
	char buffer[32];
	sprintf(buffer, "%d", counter);
	return stdm::string(buffer, alloc);
}

stdm::string ProcessHooks(hookstate_t &hooked, strings_t &stringPool, AnnotedCharRange src, Include *include, const char *const *preHooked, UINT preHookedCount, Allocator *alloc);

struct SourcePipe
{
	const AnnotedCharRange *srcRanges;
	
	const char *lineCursor;
	int line;

	const char *nextChar;
	
	SourcePipe(const AnnotedCharRange *srcRanges)
		: srcRanges(srcRanges),
		
		lineCursor(srcRanges->begin),
		line(srcRanges->annot.line),
		
		nextChar(srcRanges->begin) { }

	const char* CurrentSource()
	{
		// Find next source range containing more characters
		while (nextChar >= srcRanges->end)
		{
			++srcRanges;
			lineCursor = srcRanges->begin;
			line = srcRanges->annot.line;
		}

		return srcRanges->annot.name;
	}

	int CurrentLine()
	{
		CurrentSource();

		// Update line number for the next character
		for (; lineCursor < nextChar; ++lineCursor)
			line += is_endl(*lineCursor);

		return line;
	}

	CharRangeAnnotation CurrentAnnot()
	{
		return CharRangeAnnotation(CurrentSource(), CurrentLine());
	}

	void Flush(stdm::vector_t<AnnotedCharRange>::type &processedRanges, const char *flushEnd)
	{
		// Append source ranges up to the given character
		while (nextChar < flushEnd)
		{
			// ORDER: Update current source range before computing flush range
			const char *currentSrc = CurrentSource();
			int currentLine = CurrentLine();

			// Append the next range
			CharRange nextRange(nextChar, min(flushEnd, srcRanges->end));
			processedRanges.push_back( AnnotedCharRange(nextRange, currentSrc, currentLine) );
			nextChar = nextRange.end;
		}
	}
	
	void SkipTo(const char *flushBegin)
	{
		nextChar = flushBegin;
	}
};

AnnotedString ProcessHookIncludes(hookstate_t &hooked, strings_t &persistentStrings, AnnotedCharRange src,
								  Include *include, const char *const *preHooked, UINT preHookedCount, Allocator *alloc)
{
	stdm::vector_t<AnnotedCharRange>::type processedRanges(alloc);
	strings_t stringPool(alloc);

	SourcePipe srcPipe(&src);
	LexerState lexer(LexerCursor(src.begin, src.annot.line), src.end);

	while (dont_accept_next_token(lexer, TokenType::end))
	{
		if (!check_token(lexer, Token(TokenType::punct, "#")))
			continue;

		const char *replacementBegin = lexer.token.range.begin;

		if (accept_next_token(lexer, Token(TokenType::identifier, "hookincl"), ParserFlags::keep_endl))
		{
			if (accept_next_token(lexer, TokenType::string, ParserFlags::keep_endl))
			{
				CharRange hookedFileNameRange = deflate(lexer.accepted.range);
				const stdm::string &hookedFileName = add_by_swap( persistentStrings,
						construct_ref(stdm::string)(hookedFileNameRange.begin, hookedFileNameRange.end, alloc)
					);

				const char *const *preHookedPassed = nullptr;
				UINT preHookedPassedCount = 0;

				// Optionally pass on pre-hooked modifiers
				if (accept_next_token(lexer, Token(TokenType::punct, "..."), ParserFlags::keep_endl))
				{
					preHookedPassed = preHooked;
					preHookedPassedCount = preHookedCount;
				}

				// Open included file
				IncludeFile hookedFile(include, IncludeType::Local, hookedFileName.c_str(), src.begin);
				AnnotedCharRange hookedSrcRange(CharRange(hookedFile.src, hookedFile.srcLength), hookedFileName.c_str(), 1);
				
				// Process included file
				stdm::string hookedProcessed = ProcessHooks(hooked, persistentStrings, hookedSrcRange, include, preHookedPassed, preHookedPassedCount, alloc);

				// Insert processed content
				srcPipe.Flush(processedRanges, replacementBegin);
				processedRanges.push_back( AnnotedCharRange( add_by_swap(stringPool, hookedProcessed), hookedSrcRange.annot ) );
			}
			else if (accept_next_token(lexer, Token(TokenType::punct, "..."), ParserFlags::keep_endl))
			{
				srcPipe.Flush(processedRanges, replacementBegin);

				// Include all pre-hooked files
				for (UINT i = 0; i < preHookedCount; ++i)
				{
					IncludeFile hookedFile(include, IncludeType::Local, preHooked[i], src.begin);
					AnnotedCharRange hookedSrcRange(CharRange(hookedFile.src, hookedFile.srcLength), preHooked[i], 1);

					stdm::string hookedProcessed = ProcessHooks(hooked, persistentStrings, hookedSrcRange, include, nullptr, 0, alloc);

					processedRanges.push_back( AnnotedCharRange( add_by_swap(stringPool, hookedProcessed), hookedSrcRange.annot ) );
				}
			}
			else
				LogLineAndThrow("expected hook file or '...' in '#hookincl' statement");

			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "include"), ParserFlags::keep_endl))
		{
			while (dont_accept_next_token(lexer, TokenType::endline, ParserFlags::end_is_endl));
			
			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();

			processedRanges.push_back( AnnotedCharRange("#undef _HOOKED\n", replacementAnnot) );
			srcPipe.Flush(processedRanges, lexer.acceptedCursor.nextChar);
			processedRanges.push_back( AnnotedCharRange("\n#define _HOOKED", replacementAnnot) );
		}
	}

	srcPipe.Flush(processedRanges, lexer.acceptedCursor.nextChar);
	return merge_annoted_ranges(processedRanges, alloc);
}

AnnotedString ExpandHooks(CharRange srcChars, const AnnotedCharRange *srcRanges,
						 const hookstate_t &hooked, const char *const *preHooked, UINT preHookedCount, Allocator *alloc)
{
	stdm::vector_t<AnnotedCharRange>::type processedRanges(alloc);
	strings_t stringPool(alloc);

	SourcePipe srcPipe(srcRanges);
	LexerState lexer(LexerCursor(srcChars.begin, srcRanges->annot.line), srcChars.end);

	while (dont_accept_next_token(lexer, TokenType::end, ParserFlags::keep_endl))
	{
		const char *replacementBegin = lexer.token.range.begin;

		if (accept_next_token(lexer, Token(TokenType::identifier, "hookinsert"), ParserFlags::keep_endl))
		{
			if (!accept_next_token(lexer, TokenType::identifier, ParserFlags::keep_endl))
				LogLineAndThrow("expected hook definition identifier following '#hookinsert'");
			CharRange hookName = lexer.accepted.range;

			srcPipe.Flush(processedRanges, replacementBegin);
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			bool hookFound = false;

			for (hookstate_t::const_iterator it = hooked.begin(), itEnd = hooked.end(); it != itEnd; ++it)
			{
				const HookState &hookState = *it;

				for (size_t j = 0, defCount = hookState.definitions.size(); j < defCount; ++j)
				{
					const HookDef &def = hookState.definitions[j];

					if (def.name == hookName)
					{
						if (hookFound)
							processedRanges.push_back( AnnotedCharRange("\n\n", KeepAnnotations) );

						processedRanges.push_back( def.defs );
						hookFound = true;
					}
				}
			}
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "hookstate"), ParserFlags::keep_endl))
		{
			if (!accept_next_token(lexer, TokenType::identifier, ParserFlags::keep_endl))
				LogLineAndThrow("expected hook function/state identifier following '#hookstate'");
			CharRange hookName = lexer.accepted.range;
			
			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			bool hookFound = false;

			for (hookstate_t::const_iterator it = hooked.end(), itBegin = hooked.begin(); !hookFound && it-- != itBegin; )
			{
				const HookState &hookState = *it;

				for (size_t j = 0, funCount = hookState.functions.size(); j < funCount; ++j)
				{
					const HookFun &fun = hookState.functions[j];

					if (fun.name == hookName)
					{
						if (hookFound)
							LogLineAndThrow("'#hookcall' ambiguous, several hook functions defined");
						hookFound = true;

						processedRanges.push_back( AnnotedCharRange(fun.stateType, replacementAnnot) );
					}
				}
			}

			if (!hookFound)
				LogLineAndThrow("'#hookstate' unresolved, no matching hook functions defined");
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "hookcall"), ParserFlags::keep_endl))
		{
			if (!accept_next_token(lexer, TokenType::identifier))
				LogLineAndThrow("expected hook function/state identifier following '#hookcall'");
			CharRange hookName = lexer.accepted.range;
			
			CharRange stateName;

			if (accept_next_token(lexer, Token(TokenType::punct, "=")))
			{
				stateName = hookName;

				if (!accept_next_token(lexer, TokenType::identifier))
					LogLineAndThrow("expected hook function following state identifier + '=' in '#hookcall' statement");
				hookName = lexer.accepted.range;
			}

			if (!accept_next_token(lexer, Token(TokenType::punct, "(")))
				LogLineAndThrow("expected '(' following hook function identifier in '#hookcall' statement");

			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			bool hookFound = false;

			for (hookstate_t::const_iterator it = hooked.end(), itBegin = hooked.begin(); !hookFound && it-- != itBegin; )
			{
				const HookState &hookState = *it;

				for (size_t j = 0, funCount = hookState.functions.size(); j < funCount; ++j)
				{
					const HookFun &fun = hookState.functions[j];

					if (fun.name == hookName)
					{
						if (hookFound)
							LogLineAndThrow("'#hookcall' ambiguous, several hook functions defined");
						hookFound = true;

						if (!empty(stateName))
						{
							processedRanges.push_back( AnnotedCharRange(fun.stateType, replacementAnnot) );
							processedRanges.push_back( AnnotedCharRange(" ", KeepAnnotations) );
							processedRanges.push_back( AnnotedCharRange(stateName, KeepAnnotations) );
							processedRanges.push_back( AnnotedCharRange(" = ", KeepAnnotations) );
						}

						processedRanges.push_back( AnnotedCharRange(fun.name, replacementAnnot) );
						processedRanges.push_back( AnnotedCharRange("__HOOKFUN", KeepAnnotations) );
						processedRanges.push_back( AnnotedCharRange(hookState.hookID, KeepAnnotations)  );

						// IMPORTANT: Re-insert arguments without break
						processedRanges.push_back( AnnotedCharRange("(", KeepAnnotations) );
					}
				}
			}

			if (!hookFound)
				LogLineAndThrow("'#hookcall' unresolved, no matching hook functions defined");
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "hookstring"), ParserFlags::keep_endl))
		{
			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			processedRanges.push_back( AnnotedCharRange("\"", replacementAnnot) );

			for (UINT i = 0; i < preHookedCount; ++i)
			{
				if (i) processedRanges.push_back( AnnotedCharRange(", ", KeepAnnotations) );
				processedRanges.push_back( AnnotedCharRange(preHooked[i], KeepAnnotations) );
			}

			processedRanges.push_back( AnnotedCharRange("\"", KeepAnnotations) );
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "hookarray"), ParserFlags::keep_endl))
		{
			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			processedRanges.push_back( AnnotedCharRange("{ ", replacementAnnot) );

			for (UINT i = 0; i < preHookedCount; ++i)
			{
				if (i) processedRanges.push_back( AnnotedCharRange(", ", KeepAnnotations) );
				processedRanges.push_back( AnnotedCharRange("\"", KeepAnnotations) );
				processedRanges.push_back( AnnotedCharRange(preHooked[i], KeepAnnotations) );
				processedRanges.push_back( AnnotedCharRange("\"", KeepAnnotations) );
			}

			// NOTE: Empty arrays unsupported
			if (preHookedCount == 0)
				processedRanges.push_back( AnnotedCharRange("\"\"", KeepAnnotations) );

			processedRanges.push_back( AnnotedCharRange(" }", KeepAnnotations) );
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "hooklist"), ParserFlags::keep_endl))
		{
			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			for (UINT i = 0; i < preHookedCount; ++i)
			{
				if (i) processedRanges.push_back( AnnotedCharRange(", ", KeepAnnotations) );
				processedRanges.push_back( AnnotedCharRange("\"", replacementAnnot) );
				processedRanges.push_back( AnnotedCharRange(preHooked[i], KeepAnnotations) );
				processedRanges.push_back( AnnotedCharRange("\"", KeepAnnotations) );
			}

			// NOTE: Empty arrays unsupported
			if (preHookedCount == 0)
				processedRanges.push_back( AnnotedCharRange("\"\"", replacementAnnot) );
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "redefine"), ParserFlags::keep_endl))
		{
			if (!accept_next_token(lexer, TokenType::identifier, ParserFlags::keep_endl))
				LogLineAndThrow("expected identifier following '#redefine'");
			CharRange name = lexer.accepted.range;

			while (dont_accept_next_token(lexer, TokenType::endline, ParserFlags::end_is_endl));
			CharRange decl(name.end, lexer.accepted.range.end);

			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			processedRanges.push_back( AnnotedCharRange("#ifndef ", replacementAnnot) );
			processedRanges.push_back( AnnotedCharRange(name, KeepAnnotations) );
			processedRanges.push_back( AnnotedCharRange("\n#define ", replacementAnnot) );
			processedRanges.push_back( AnnotedCharRange(CharRange(name.begin, decl.end), KeepAnnotations) );
			processedRanges.push_back( AnnotedCharRange("\n#endif", replacementAnnot) );
		}
	}

	srcPipe.Flush(processedRanges, lexer.acceptedCursor.nextChar);
	return merge_annoted_ranges(processedRanges, alloc);
}

AnnotedString BuildHookStateAndStrip(HookState &hook, CharRange srcChars, const AnnotedCharRange *srcRanges, Allocator *alloc)
{
	stdm::vector_t<AnnotedCharRange>::type processedRanges(alloc);
	strings_t stringPool(alloc);

	SourcePipe srcPipe(srcRanges);
	LexerState lexer(LexerCursor(srcChars.begin, srcRanges->annot.line), srcChars.end);

	while (dont_accept_next_token(lexer, TokenType::end))
	{
		if (!check_token(lexer, Token(TokenType::punct, "#")))
			continue;

		const char *replacementBegin = lexer.token.range.begin;

		if (accept_next_token(lexer, Token(TokenType::identifier, "hookdef"), ParserFlags::keep_endl))
		{
			HookDef hookDef;

			if (!accept_next_token(lexer, TokenType::identifier, ParserFlags::keep_endl))
				LogLineAndThrow("expected hook definition identifier following '#hookdef'");
			hookDef.name = lexer.accepted.range;

			if (!accept_next_token(lexer, Token(TokenType::punct, "{")))
				LogLineAndThrow("expected '{' following '#hookdef' declaration");

			// NOTE: Exclude opening braces
			hookDef.defs.begin = hookDef.defs.end = next_token(lexer).range.begin;
			
			// Skip to hook definition & store corresponding annotation
			srcPipe.Flush(processedRanges, replacementBegin);
			srcPipe.SkipTo(hookDef.defs.begin);
			hookDef.defs.annot = srcPipe.CurrentAnnot();

			for (int openCounter = 1; openCounter > 0; )
			{
				if (!dont_accept_next_token(lexer, TokenType::end))
					LogLineAndThrow("expected '}' matching '#hookdef' declaration");

				if (check_token(lexer, Token(TokenType::punct, "}")))
					--openCounter;
				else if (check_token(lexer, Token(TokenType::punct, "{")))
					++openCounter;

				// NOTE: Exclude closing braces
				if (openCounter > 0)
					hookDef.defs.end = lexer.accepted.range.end;
			}

			// NOTE: Exclude closing braces
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			hook.definitions.push_back(hookDef);
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "hookfun"), ParserFlags::keep_endl))
		{
			HookFun hookFun;

			if (!accept_next_token(lexer, TokenType::identifier, ParserFlags::keep_endl))
				LogLineAndThrow("expected hook function identifier following '#hookfun'");
			hookFun.name = lexer.accepted.range;
			
			if (!accept_next_token(lexer, Token(TokenType::punct, "(")))
				LogLineAndThrow("expected '(' following '#hookfun' declaration");
			
			hookFun.decl = lexer.accepted.range;

			while (!accept_next_token(lexer, Token(TokenType::punct, ")")))
			{
				if (!dont_accept_next_token(lexer, TokenType::end))
					LogLineAndThrow("expected ')' matching '#hookfun' declaration");
			}

			hookFun.decl.end = lexer.accepted.range.end;

			if (!accept_next_token(lexer, Token(TokenType::punct, "->")) ||
				!accept_next_token(lexer, TokenType::identifier))
				LogLineAndThrow("expected '->' + return type following '#hookfun' declaration");
			hookFun.stateType = lexer.accepted.range;

			if (!accept_next_token(lexer, Token(TokenType::punct, "=")))
				LogLineAndThrow("expected '=' following '#hookfun' declaration");

			if (!dont_accept_next_token(lexer, TokenType::endline, ParserFlags::end_is_endl))
				LogLineAndThrow("expected *something* following '=' in '#hookfun' declaration");
			hookFun.call = lexer.accepted.range;
			while (dont_accept_next_token(lexer, TokenType::endline, ParserFlags::end_is_endl));
			hookFun.call.end = lexer.accepted.range.end;

			// NOTE: Exclude endl
			srcPipe.Flush(processedRanges, replacementBegin);
			CharRangeAnnotation replacementAnnot = srcPipe.CurrentAnnot();
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);

			hook.functions.push_back(hookFun);

			// Build macro code
			processedRanges.push_back( AnnotedCharRange("#define ", replacementAnnot) );
			processedRanges.push_back( AnnotedCharRange(hookFun.name, KeepAnnotations) );
			processedRanges.push_back( AnnotedCharRange("__HOOKFUN", KeepAnnotations));
			processedRanges.push_back( AnnotedCharRange(hook.hookID, KeepAnnotations) );
			processedRanges.push_back( AnnotedCharRange(hookFun.decl, KeepAnnotations));
			processedRanges.push_back( AnnotedCharRange(" ", KeepAnnotations) );
			processedRanges.push_back( AnnotedCharRange(hookFun.call, KeepAnnotations) );
		}
		else if (accept_next_token(lexer, Token(TokenType::identifier, "hookconfig"), ParserFlags::keep_endl))
		{
			if (!accept_next_token(lexer, Token(TokenType::punct, "{")))
				LogLineAndThrow("expected '{' following '#hookdef' declaration");

			// TODO: Store annotated range
			hook.configuration = CharRange(next_token(lexer).range.begin, (size_t) 0);

			for (int openCounter = 1; openCounter > 0; )
			{
				if (!dont_accept_next_token(lexer, TokenType::end))
					LogLineAndThrow("expected '}' matching '#hookconfig' declaration");

				if (check_token(lexer, Token(TokenType::punct, "}")))
					--openCounter;
				else if (check_token(lexer, Token(TokenType::punct, "{")))
					++openCounter;

				// NOTE: Exclude closing braces
				if (openCounter > 0)
					hook.configuration.end = lexer.accepted.range.end;
			}

			// NOTE: Exclude closing braces
			srcPipe.Flush(processedRanges, replacementBegin);
			srcPipe.SkipTo(lexer.acceptedCursor.nextChar);
		}
	}

	srcPipe.Flush(processedRanges, lexer.acceptedCursor.nextChar);
	return merge_annoted_ranges(processedRanges, alloc);
}

stdm::string ProcessHooks(hookstate_t &hooked, strings_t &stringPool, AnnotedCharRange src, Include *include, const char *const *preHooked, UINT preHookedCount, Allocator *alloc)
{
	AnnotedString included = ProcessHookIncludes(hooked, stringPool, src, include, preHooked, preHookedCount, alloc);
	
	AnnotedString expanded = ExpandHooks(CharRange(included.text), &included.annotations[0], hooked, preHooked, preHookedCount, alloc);

	hooked.push_back(HookState(alloc));
	HookState &hookState = hooked.back();
	hookState.hookID = MakeHookID((UINT) hooked.size(), alloc);
	
	AnnotedString finalAnnoted = BuildHookStateAndStrip(hookState, CharRange(add_by_swap(stringPool, expanded.text)), &expanded.annotations[0], alloc);
	return fix_lines(&finalAnnoted.annotations[0], &finalAnnoted.annotations[0] + finalAnnoted.annotations.size(), alloc);
}

stdm::string AddPreamble(const hookstate_t &hooked, CharRange src, Allocator *alloc, char const *pCustomPreamble)
{
	stdm::vector_t<CharRange>::type processedRanges(alloc);

	if (pCustomPreamble)
	{
		processedRanges.push_back(pCustomPreamble);
		processedRanges.push_back("\n\n");
	}

	processedRanges.push_back("#define _HOOKED");

	for (hookstate_t::const_iterator it = hooked.begin(), itEnd = hooked.end(); it != itEnd; ++it)
	{
		if (!empty(it->configuration))
		{
			processedRanges.push_back("\n\n");
			processedRanges.push_back(it->configuration);
			processedRanges.push_back("\n\n");
		}
	}

	processedRanges.push_back("\n\n");
	processedRanges.push_back(src);
	
	return merge_ranges(processedRanges, alloc);
}

} // namespace

// Resolves hooks in the given uncompiled effect.
Blob* D3DEFFECTSLITE_STDCALL HookEffect(const void *bytes, UINT byteCount, Include *include, const char *const *hooked, UINT hookedCount,
										const char *srcName, const char *pCustomPreamble,
										Allocator *pScratchAllocator)
{
	assert(bytes);
	assert(byteCount);

	if (!srcName) srcName = "unknown";

	if (!include)
	{
		D3DEFFECTSLITE_LOG_LINE("include manager may not be null for hooking");
		return nullptr;
	}

	if (!pScratchAllocator)
		pScratchAllocator = GetGlobalAllocator();

	com_ptr<Blob> result;

	try
	{
		hookstate_t hookState(pScratchAllocator);
		strings_t stringPool(pScratchAllocator);

		AnnotedCharRange srcRange(
			CharRange(static_cast<const char*>(bytes), byteCount),
			srcName,
			1
		);

		stdm::string processed = ProcessHooks(hookState, stringPool, srcRange, include, hooked, hookedCount, pScratchAllocator);
		stdm::string complete = AddPreamble(hookState, CharRange(processed), pScratchAllocator, pCustomPreamble);
		result = CreateBlob(complete.data(), (UINT) complete.size());
	}
	catch (const LoggedError&)
	{
		// NOTE: Already reported
	}
	catch (...)
	{
		D3DEFFECTSLITE_LOG_LINE("Unknown error");
	}

	return result.unbind();
}

} // namespace

// Resolves hooks in the given uncompiled effect.
D3DEffectsLiteBlob* D3DEFFECTSLITE_STDCALL D3DELHookEffect(const void *bytes, UINT byteCount,
														   D3DEffectsLiteInclude *include, const char *const *hooked, UINT hookedCount,
														   const char *srcName, const char *pCustomPreamble,
														   D3DEffectsLiteAllocator *pScratchAllocator)
{
	return D3DEffectsLite::HookEffect(bytes, byteCount, include, hooked, hookedCount, srcName, pCustomPreamble, pScratchAllocator);
}

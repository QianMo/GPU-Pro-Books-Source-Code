/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/


/******************************************************************************
 * Type traits for numeric types
 ******************************************************************************/

#pragma once

#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


enum Representation
{
	NOT_A_NUMBER,
	SIGNED_INTEGER,
	UNSIGNED_INTEGER,
	FLOATING_POINT
};


template <Representation R>
struct BaseTraits
{
	enum {
		REPRESENTATION 		= R,
		BUILT_IN			= (R != NOT_A_NUMBER),
	};
};


// Default, non-numeric types
template <typename T> struct CleanedTraits : 				BaseTraits<NOT_A_NUMBER> {};

template <> struct CleanedTraits<char> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct CleanedTraits<signed char> : 			BaseTraits<SIGNED_INTEGER> {};
template <> struct CleanedTraits<short> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct CleanedTraits<int> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct CleanedTraits<long> : 					BaseTraits<SIGNED_INTEGER> {};
template <> struct CleanedTraits<long long> : 				BaseTraits<SIGNED_INTEGER> {};

template <> struct CleanedTraits<unsigned char> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct CleanedTraits<unsigned short> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct CleanedTraits<unsigned int> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct CleanedTraits<unsigned long> : 			BaseTraits<UNSIGNED_INTEGER> {};
template <> struct CleanedTraits<unsigned long long> : 		BaseTraits<UNSIGNED_INTEGER> {};

template <> struct CleanedTraits<float> : 					BaseTraits<FLOATING_POINT> {};
template <> struct CleanedTraits<double> : 					BaseTraits<FLOATING_POINT> {};

/**
 * Numeric traits.
 *
 * Removes any volatile / const qualifiers
 */
template <typename T>
struct NumericTraits : CleanedTraits<typename RemoveQualifiers<T>::Type> {};


} // namespace util
} // namespace b40c
B40C_NS_POSTFIX

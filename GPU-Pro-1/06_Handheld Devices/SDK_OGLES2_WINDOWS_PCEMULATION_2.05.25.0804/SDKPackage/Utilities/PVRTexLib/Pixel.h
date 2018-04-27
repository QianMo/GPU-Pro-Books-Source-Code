/******************************************************************************

 @File         Pixel.h

 @Title        Console Log

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI

 @Description  Pixel structs.

******************************************************************************/
#ifndef PIXEL_H
#define PIXEL_H

#include <assert.h>

namespace pvrtexlib
{
	
#ifdef __APPLE__
	/* The classes below are exported */
#pragma GCC visibility push(default)
#endif
	
	// note order of enums reflecta order of members of Pixel struct
	enum E_COLOUR_CHANNEL
	{
		ePIXEL_Red=0,
		ePIXEL_Green,
		ePIXEL_Blue,
		ePIXEL_Alpha,
		ePIXEL_None
	};

	// Pixel structures for standard pixel types
	template<typename channelType> struct Pixel
	{
		typedef channelType chanType;
		channelType Red, Green, Blue, Alpha;

		Pixel(){}
		Pixel(unsigned int u32Colour);

		Pixel(channelType nRed, channelType nGreen, channelType nBlue, channelType nAlpha)
			:Red(nRed),Green(nGreen),Blue(nBlue),Alpha(nAlpha){}

		unsigned int	toUnsignedInt();
		inline channelType		getMaxChannelValue() const;
		inline int				getSizeInBytes() const;

		channelType& operator[](const unsigned int channel)
		{
			assert(channel<4);
			return ((channelType*)(&Red))[channel];
		}

		bool operator==(const Pixel& b)
		{
			return Red==b.Red && Green==b.Green && Blue==b.Blue && Alpha==b.Alpha;
		}


		Pixel<channelType> operator+(const Pixel& b)
		{
			return Pixel<channelType>(Red+b.Red,Green+b.Green,Blue+b.Blue,Alpha+b.Alpha);
		}
		Pixel<channelType>& operator+=(const Pixel<channelType>& b)
		{
			Red +=b.Red;Green+=b.Green;Blue+=b.Blue;Alpha+=b.Alpha;
			return *this;
		}
		Pixel<channelType>& operator+=(const channelType b)
		{
			Red +=b;Green+=b;Blue+=b;Alpha+=b;
			return *this;
		}

		Pixel<channelType> operator-(const Pixel& b)
		{
			return Pixel<channelType>(Red-b.Red,Green-b.Green,Blue-b.Blue,Alpha-b.Alpha);
		}

		Pixel<channelType> operator*(const Pixel<channelType>& b)
		{
			return Pixel<channelType>(Red*b.Red,Green*b.Green,Blue*b.Blue,Alpha*b.Alpha);
		}
		Pixel<channelType> operator*(const channelType b)
		{
			return Pixel<channelType>(Red*b,Green*b,Blue*b,Alpha*b);
		}

		Pixel<channelType>& operator*=(const channelType b)
		{
			Red *=b;Green*=b;Blue*=b;Alpha*=b;
			return *this;
		}

		Pixel<channelType> operator/(const Pixel<channelType>& b)
		{
			return Pixel<channelType>(Red/b.Red,Green/b.Green,Blue/b.Blue,Alpha/b.Alpha);
		}

		friend inline	Pixel<channelType> operator*(const float f,const Pixel<channelType>& b)
		{
			return Pixel<channelType>((channelType)(f*(float)b.Red),
				(channelType)(f*(float)b.Green),
				(channelType)(f*(float)b.Blue),
				(channelType)(f*(float)b.Alpha));
		}

	/*******************************************************************************
	* Constructor
	* Description		: Copy constructor
	*******************************************************************************/
		template<typename T>
	Pixel(const Pixel<T>& original)
	{
		Red = (channelType)original.Red;
		Green = (channelType)original.Green;
		Blue = (channelType)original.Blue;
		Alpha = (channelType)original.Alpha;
	}
	/*******************************************************************************
	* Constructor
	* Description		: Assignment operator
	*******************************************************************************/
		template<typename T>
	Pixel& operator=(const Pixel<T>& sRHS)
	{
		Red = (channelType)sRHS.Red;
		Green = (channelType)sRHS.Green;
		Blue = (channelType)sRHS.Blue;
		Alpha = (channelType)sRHS.Alpha;
		return *this;
	}

	};

	template<typename channelType>
	inline int Pixel<channelType>::getSizeInBytes() const
	{
		return sizeof(channelType)*4;
	}


	// specialisations for uint8
	template<>
	inline Pixel<uint8>::Pixel(const unsigned int u32Colour)
	{
		Alpha = uint8(u32Colour>>24);
		Blue = uint8((u32Colour>>16)&0xff);
		Green = uint8((u32Colour>>8)&0xff);
		Red = uint8(u32Colour&0xff);
	}
	template<>
	inline unsigned int	Pixel<uint8>::toUnsignedInt()
	{
		return ((unsigned int)(Alpha)<<24)
			|((unsigned int) (Blue)<<16)
			|((unsigned int) (Green)<<8)
			|(unsigned int) (Red);
	}

	template<>
	inline uint8		Pixel<uint8>::getMaxChannelValue() const
	{
		return 0xFF;
	}

	// specialisations for uint16
	template<>
	inline uint16		Pixel<uint16>::getMaxChannelValue() const
	{
		return 0xFFFF;
	}
	template<>
	inline unsigned int	Pixel<uint16>::toUnsignedInt()
	{
		return ((unsigned int)(Alpha>>8)<<24)
			|((unsigned int) (Blue>>8)<<16)
			|((unsigned int) (Green>>8)<<8)
			|(unsigned int) (Red>>8);
	}

	// specialisations for uint32
	template<>
	inline uint32		Pixel<uint32>::getMaxChannelValue() const
	{
		return 0xFFFFFFFF;
	}
	template<>
	inline unsigned int	Pixel<uint32>::toUnsignedInt()
	{
		return ((unsigned int)(Alpha>>24)<<24)
			|((unsigned int) (Blue>>24)<<16)
			|((unsigned int) (Green>>24)<<8)
			|(unsigned int) (Red>>24);
	}

	// specialisations for float
	template<>
	inline float32		Pixel<float32>::getMaxChannelValue() const
	{
		return 1.0f;
	}
	template<>
	inline unsigned int	Pixel<float32>::toUnsignedInt()
	{
		return ((unsigned int)(Alpha*255.f)<<24)
			|((unsigned int) (Blue*255.f)<<16)
			|((unsigned int) (Green*255.f)<<8)
			|(unsigned int) (Red*255.f);
	}

	// convenience struct for PixelFormats
	template<typename channelType> struct PixelRGB
	{
		channelType Red;
		channelType Green;
		channelType Blue;

		// constructors
		PixelRGB(){}
		PixelRGB(const channelType tRed,
			const channelType tGreen,
			const channelType tBlue):
			Red(tRed),
			Green(tGreen),
			Blue(tBlue){}
	};

	// convenience struct for PixelFormats
	template<typename channelType> struct PixelRG
	{
		channelType Red;
		channelType Green;
		PixelRG(){}
		PixelRG(const channelType tRed,
			const channelType tGreen):
			Red(tRed),
			Green(tGreen){}
	};

#ifdef __APPLE__
#pragma GCC visibility pop
#endif
}
#endif // PIXEL_H

/*****************************************************************************
End of file (Pixel.h)
*****************************************************************************/

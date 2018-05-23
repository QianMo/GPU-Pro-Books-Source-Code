// image.c - public domain APNG loader / exporter for OpenGL
// author: Jonathan Dupuy (jdupuy at liris.cnrs.fr)
#define PNG_NO_WRITE_tIME 1
#ifdef _WIN32
#	include "png.h"
#	include "zlib.h"
#else
#	include <png.h>
#	include <zlib.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include "bstrlib.h"
#include "glload.h"
#include "image.h"

// utility macros
#define IMAGE_MAX2(x,y  ) (x) > (y) ? (x) : (y)
#define IMAGE_MAX3(x,y,z) IMAGE_MAX2(IMAGE_MAX2(x,y), z)

// image type
struct image_t {
	GLubyte *data;
	GLint width, height, depth;
	GLenum format;
	GLenum type;
};

//---------------------------------------------------------------------------
// misc tools
#if GL_ARB_texture_storage
static int
image_next_pot (int x) {
	size_t i;

	--x;
	for (i = 1; i < sizeof (int) * 8; i <<= 1) {
		x = x | x >> i;
	}
	return (x + 1);
}

static int
image_pot_exp (int x) {
	int p = image_next_pot (x);
	int e = 0;

	while (! (p & 1)) {
		p >>= 1;
		++e;
	}
	return e;
}
#endif

//---------------------------------------------------------------------------
// OpenGL pixel store state manipulation
typedef GLint image_gl_pixel_store_state_t [9];

static void
image_get_pixel_pack_state (image_gl_pixel_store_state_t state) {
	glGetIntegerv (GL_PIXEL_PACK_BUFFER_BINDING, state    );
	glGetIntegerv (GL_PACK_SWAP_BYTES          , state + 1);
	glGetIntegerv (GL_PACK_LSB_FIRST           , state + 2);
	glGetIntegerv (GL_PACK_ROW_LENGTH          , state + 3);
	glGetIntegerv (GL_PACK_IMAGE_HEIGHT        , state + 4);
	glGetIntegerv (GL_PACK_SKIP_ROWS           , state + 5);
	glGetIntegerv (GL_PACK_SKIP_PIXELS         , state + 6);
	glGetIntegerv (GL_PACK_SKIP_IMAGES         , state + 7);
	glGetIntegerv (GL_PACK_ALIGNMENT           , state + 8);
}

static void
image_get_pixel_unpack_state (image_gl_pixel_store_state_t state) {
	glGetIntegerv (GL_PIXEL_UNPACK_BUFFER_BINDING, state    );
	glGetIntegerv (GL_UNPACK_SWAP_BYTES          , state + 1);
	glGetIntegerv (GL_UNPACK_LSB_FIRST           , state + 2);
	glGetIntegerv (GL_UNPACK_ROW_LENGTH          , state + 3);
	glGetIntegerv (GL_UNPACK_IMAGE_HEIGHT        , state + 4);
	glGetIntegerv (GL_UNPACK_SKIP_ROWS           , state + 5);
	glGetIntegerv (GL_UNPACK_SKIP_PIXELS         , state + 6);
	glGetIntegerv (GL_UNPACK_SKIP_IMAGES         , state + 7);
	glGetIntegerv (GL_UNPACK_ALIGNMENT           , state + 8);
}

static void
image_set_pixel_pack_state ( const image_gl_pixel_store_state_t state) {
	glBindBuffer (GL_PIXEL_PACK_BUFFER , state[0]);
	glPixelStorei (GL_PACK_SWAP_BYTES  , state[1]);
	glPixelStorei (GL_PACK_LSB_FIRST   , state[2]);
	glPixelStorei (GL_PACK_ROW_LENGTH  , state[3]);
	glPixelStorei (GL_PACK_IMAGE_HEIGHT, state[4]);
	glPixelStorei (GL_PACK_SKIP_ROWS   , state[5]);
	glPixelStorei (GL_PACK_SKIP_PIXELS , state[6]);
	glPixelStorei (GL_PACK_SKIP_IMAGES , state[7]);
	glPixelStorei (GL_PACK_ALIGNMENT   , state[8]);
}

static void
image_set_pixel_unpack_state ( const image_gl_pixel_store_state_t state) {
	glBindBuffer (GL_PIXEL_UNPACK_BUFFER , state[0]);
	glPixelStorei (GL_UNPACK_SWAP_BYTES  , state[1]);
	glPixelStorei (GL_UNPACK_LSB_FIRST   , state[2]);
	glPixelStorei (GL_UNPACK_ROW_LENGTH  , state[3]);
	glPixelStorei (GL_UNPACK_IMAGE_HEIGHT, state[4]);
	glPixelStorei (GL_UNPACK_SKIP_ROWS   , state[5]);
	glPixelStorei (GL_UNPACK_SKIP_PIXELS , state[6]);
	glPixelStorei (GL_UNPACK_SKIP_IMAGES , state[7]);
	glPixelStorei (GL_UNPACK_ALIGNMENT   , state[8]);
}

static void
image_default_pixel_pack_state () {
	glBindBuffer (GL_PIXEL_PACK_BUFFER , 0);
	glPixelStorei (GL_PACK_SWAP_BYTES  , 0);
	glPixelStorei (GL_PACK_LSB_FIRST   , 0);
	glPixelStorei (GL_PACK_ROW_LENGTH  , 0);
	glPixelStorei (GL_PACK_IMAGE_HEIGHT, 0);
	glPixelStorei (GL_PACK_SKIP_ROWS   , 0);
	glPixelStorei (GL_PACK_SKIP_PIXELS , 0);
	glPixelStorei (GL_PACK_SKIP_IMAGES , 0);
	glPixelStorei (GL_PACK_ALIGNMENT   , 2);
}

static void
image_default_pixel_unpack_state () {
	glBindBuffer (GL_PIXEL_UNPACK_BUFFER , 0);
	glPixelStorei (GL_UNPACK_SWAP_BYTES  , 0);
	glPixelStorei (GL_UNPACK_LSB_FIRST   , 0);
	glPixelStorei (GL_UNPACK_ROW_LENGTH  , 0);
	glPixelStorei (GL_UNPACK_IMAGE_HEIGHT, 0);
	glPixelStorei (GL_UNPACK_SKIP_ROWS   , 0);
	glPixelStorei (GL_UNPACK_SKIP_PIXELS , 0);
	glPixelStorei (GL_UNPACK_SKIP_IMAGES , 0);
	glPixelStorei (GL_UNPACK_ALIGNMENT   , 2);
}


//---------------------------------------------------------------------------
// private queries

// num channels
static int
image_nc (const struct image_t *image) {
	if (image->format == GL_RED ) return 1;
	if (image->format == GL_RG  ) return 2;
	if (image->format == GL_RGB ) return 3;
	if (image->format == GL_RGBA) return 4;

	fprintf (stderr, "image: number of channels unknown\n");
	return -1; // error
}

// bytes per pixel
static int
image_bytespp (const struct image_t *image) {
	switch (image->type) {
		case GL_UNSIGNED_BYTE:
		case GL_BYTE:
			return (image_nc (image));
		case GL_UNSIGNED_BYTE_3_3_2:
		case GL_UNSIGNED_BYTE_2_3_3_REV:
			return (1);
		case GL_SHORT:
		case GL_UNSIGNED_SHORT:
			return (2 * image_nc (image));
		case GL_UNSIGNED_SHORT_5_6_5:
		case GL_UNSIGNED_SHORT_5_6_5_REV:
		case GL_UNSIGNED_SHORT_4_4_4_4:
		case GL_UNSIGNED_SHORT_4_4_4_4_REV:
		case GL_UNSIGNED_SHORT_5_5_5_1:
		case GL_UNSIGNED_SHORT_1_5_5_5_REV:
			return (2);
		case GL_UNSIGNED_INT:
		case GL_INT:
			return (4 * image_nc (image));
		case GL_UNSIGNED_INT_8_8_8_8:
		case GL_UNSIGNED_INT_8_8_8_8_REV:
		case GL_UNSIGNED_INT_10_10_10_2:
		case GL_UNSIGNED_INT_2_10_10_10_REV:
			return 4;
		default: 
			fprintf (stderr, "image: bytes per pixel unknown\n");
			break;
	}
	return -1; // error
}


//---------------------------------------------------------------------------
// interface for libpng

#if 0
static int
image_is_png_ready (const struct image_t *image) {
	int c1 = image->type == GL_UNSIGNED_BYTE;
	int c2 = image->type == GL_UNSIGNED_SHORT;

	return (c1 || c2);
}
#endif

static int
image_png_color_type (const struct image_t *image) {
	if (image->format == GL_RED ) return PNG_COLOR_TYPE_GRAY;
	if (image->format == GL_RG  ) return PNG_COLOR_TYPE_GRAY_ALPHA;
	if (image->format == GL_RGB ) return PNG_COLOR_TYPE_RGB;
	if (image->format == GL_RGBA) return PNG_COLOR_TYPE_RGBA;
	return -1; // error
}

static int
image_write_png (const struct image_t *image, const char *filename) {
	FILE * pf = fopen(filename, "wb");
	png_structp  png_ptr;
	png_infop    info_ptr;
	png_bytepp   row_bytes;
	int i, nc, bpp;

	if (!pf) {
		fprintf (stderr, "image: failed to open file %s\n", filename);
		return IMAGE_FAILURE;
	}

	png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) {
		fprintf (stderr, "image: png_create_write_struct failed\n");
		return IMAGE_FAILURE;
	}

	info_ptr = png_create_info_struct (png_ptr);
	if (!info_ptr) {
		fprintf (stderr, "image: png_create_info_struct failed\n");
		png_destroy_write_struct (&png_ptr, NULL);
		return IMAGE_FAILURE;
	}

	nc  = image_nc (image);
	bpp = image_bytespp (image);
	png_init_io (png_ptr, pf);
	png_set_compression_level (png_ptr, 9);
	png_set_swap (png_ptr);
	png_set_IHDR (png_ptr, info_ptr,
	              image->width, image->height,
	              (bpp / nc) << 3,
	              image_png_color_type (image),
	              0, 0, 0);

	row_bytes = malloc (sizeof (png_bytep) * image->height);

	if (!row_bytes) {
		fprintf (stderr, "image: malloc failed\n");
		png_destroy_write_struct (&png_ptr, &info_ptr);
		return IMAGE_FAILURE;
	}

	for (i = 0; i < image->height; ++i)
		row_bytes[image->height - i - 1] =
			(png_byte *) image->data + i * image->width * bpp;

	png_write_info (png_ptr, info_ptr);
	png_write_image (png_ptr, row_bytes);
	png_write_end (png_ptr, info_ptr);

	png_destroy_write_struct (&png_ptr, &info_ptr);
	fclose(pf);

	return IMAGE_SUCCESS;
}


#if 1
static struct image_t *
image_read_png (const char *filename) {
	struct image_t *image = malloc (sizeof (*image));
	FILE *fp = NULL;
	png_byte magic[8];
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep *row_pointers = NULL;
	png_uint_32 w, h;
	int row_bytes, i, bit_depth, color_type;

	if (!image) {
		fprintf (stderr, "image: malloc failed\n");
		return NULL;
	}

	fp = fopen (filename, "rb");
	if (!fp) {
		fprintf (stderr, "image: failed to open \"%s\"\n", filename);
		free (image);
		return NULL;
	}

	// read magic number
	if (fread (magic, 1, sizeof(magic), fp) != sizeof(magic)) {
		fprintf (stderr, "image: fread failed on \"%s\"\n", filename);
		free (image);
		return NULL;
	}

	// check for valid magic number
	if (!png_check_sig (magic, sizeof (magic))) {
		fprintf (stderr, "image: \"%s\" is not a valid PNG image\n", filename);
		fclose (fp);
		free (image);
		return NULL;
	}

	// create a png read struct
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) {
		fclose (fp);
		free (image);
		return NULL;
	}

	// create a png info struct
	info_ptr = png_create_info_struct (png_ptr);
	if (!info_ptr) {
		fclose (fp);
		png_destroy_read_struct (&png_ptr, NULL, NULL);
		free (image);
		return NULL;
	}

	/* Initialize the setjmp for returning properly after a libpng
	 error occured */
	if (setjmp (png_jmpbuf (png_ptr))) {
		fclose (fp);
		png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
		free (image);
		return NULL;
	}

	/* setup libpng for using standard C fread() function
	 with our FILE pointer */
	png_init_io (png_ptr, fp);

	// tell libpng that we have already read the magic number
	png_set_sig_bytes (png_ptr, sizeof(magic));

	// read png info
	png_read_info (png_ptr, info_ptr);

	// retrieve updated information
	png_get_IHDR (png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, 
	              NULL, NULL, NULL);

	// set format
	switch (color_type) {
		case PNG_COLOR_TYPE_GRAY:
#if LINUX // TODO compilation issue in windows, needs fixing !
			png_set_gray_1_2_4_to_8 (png_ptr);
#endif
			image->format = GL_RED;
			break;
		case PNG_COLOR_TYPE_PALETTE:
			png_set_palette_to_rgb (png_ptr);
			image->format = GL_RGB;
			break;
		case PNG_COLOR_TYPE_GA:
			image->format = GL_RG;
			break;
		case PNG_COLOR_TYPE_RGB:
			image->format = GL_RGB;
			break;
		case PNG_COLOR_TYPE_RGBA:
			image->format = GL_RGBA;
			break;
		default:
			fprintf(stderr, "image: unsupported pixel format\n");
			fclose (fp);
			png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
			free (image);
			return NULL;
	}

	// swap bytes for 16bit data
	png_set_swap (png_ptr);

	// set information and type
	image->width  = w;
	image->height = h;
	image->depth  = 1;
	image->type   = bit_depth == 8 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT;

	// update info structure to apply transformations
	png_read_update_info (png_ptr, info_ptr);

	// allocate memory for storing pixel data
	row_bytes = png_get_rowbytes (png_ptr, info_ptr);
	image->data = malloc (h * row_bytes);
	if (!image->data) {
		fprintf (stderr, "image: malloc failed\n");
		fclose (fp);
		png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
		free (image);
		return NULL;
	}

	// setup pointer array
	row_pointers = malloc (sizeof (png_bytep) * row_bytes);
	if (!row_pointers) {
		fprintf (stderr, "image: malloc failed\n");
		fclose (fp);
		png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
		image_release (image);
		return NULL;
	}
	for (i = 0; i < image->height; ++i) {
		row_pointers[image->height-1-i] = image->data + i * row_bytes;
	}

	// read pixels
	png_read_image (png_ptr, row_pointers);

	// finish decompression and release memory
	png_read_end (png_ptr, NULL);
	png_destroy_read_struct (&png_ptr, &info_ptr, NULL);

	// cleanup
	free (row_pointers);
	fclose (fp);

	return image;
}
#else // TODO
static struct image_t *
image_read_apng (const char *filename) {
	FILE * f1;
	png_byte apng_chunks[]= {"acTL\0fcTL\0fdAT\0"};

  if ((f1 = fopen(filename, "rb")) != 0)
  {
    png_colorp      palette;
    png_color_16p   trans_color;
    png_bytep       trans_alpha;
    unsigned int    rowbytes, j;
    unsigned char   sig[8];
    image_info      img;

    memset(&img, 0, sizeof(img));
    memset(img.tr, 255, 256);
    img.zstream.zalloc  = Z_NULL;
    img.zstream.zfree   = Z_NULL;
    img.zstream.opaque  = Z_NULL;
    inflateInit(&img.zstream);

    if (fread(sig, 1, 8, f1) == 8 && png_sig_cmp(sig, 0, 8) == 0)
    {
      png_structp png_ptr  = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      png_infop   info_ptr = png_create_info_struct(png_ptr);
      if (png_ptr != NULL && info_ptr != NULL && setjmp(png_jmpbuf(png_ptr)) == 0)
      {
        png_set_keep_unknown_chunks(png_ptr, 2, apng_chunks, 3);
        png_set_read_user_chunk_fn(png_ptr, &img, handle_apng_chunks);
        png_init_io(png_ptr, f1);
        png_set_sig_bytes(png_ptr, 8);
        png_read_info(png_ptr, info_ptr);

        img.w    = png_get_image_width(png_ptr, info_ptr);
        img.h    = png_get_image_height(png_ptr, info_ptr);
        img.d    = png_get_bit_depth(png_ptr, info_ptr);
        img.t    = png_get_color_type(png_ptr, info_ptr);
        img.ch   = png_get_channels(png_ptr, info_ptr);
        img.it   = png_get_interlace_type(png_ptr, info_ptr);
        rowbytes = png_get_rowbytes(png_ptr, info_ptr);
        printf(" IN: %s : %dx%d\n", szIn, img.w, img.h);
        img.buf_size = img.h*(rowbytes+1);
        img.buf  = (unsigned char *)malloc(img.buf_size);

        if (png_get_PLTE(png_ptr, info_ptr, &palette, &img.ps))
          memcpy(img.pl, palette, img.ps * 3);
        else
          img.ps = 0;

        if (png_get_tRNS(png_ptr, info_ptr, &trans_alpha, &img.ts, &trans_color))
        {
          if (img.t == PNG_COLOR_TYPE_PALETTE)
            memcpy(img.tr, trans_alpha, img.ts);
          else
            memcpy(&img.tc, trans_color, sizeof(png_color_16));
        }
        else
          img.ts = 0;

        (void)png_set_interlace_handling(png_ptr);
        png_read_update_info(png_ptr, info_ptr);

        img.size = img.h*rowbytes;
        img.frame = (png_bytep)malloc(img.size);
        img.rows = (png_bytepp)malloc(img.h*sizeof(png_bytep));

        if (img.buf && img.frame && img.rows)
        {
          for (j=0; j<img.h; j++)
            img.rows[j] = img.frame + j*rowbytes;

          png_read_image(png_ptr, img.rows);
          SavePNG(&img);
          png_read_end(png_ptr, info_ptr);
          free(img.rows);
          free(img.frame);
          free(img.buf);
        }
      }
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    }
    inflateEnd(&img.zstream);
    fclose(f1);
  }

	return IMAGE_SUCCESS;
}
#endif

//---------------------------------------------------------------------------
// Loaders

// load memory (data is copied)
struct image_t *
image_load_memory (int width, int height, int depth,
                   GLenum format, GLenum type,
                   const GLvoid *data) {
	struct image_t *image = malloc (sizeof (*image));
	size_t byte_count;

	if (!image) {
		fprintf (stderr, "image: malloc failed\n");
		return NULL;
	}

	image->width  = width;
	image->height = height;
	image->depth  = depth;
	image->format = format;
	image->type   = type;
	byte_count = width * height * depth * image_bytespp (image);
	image->data   = malloc (byte_count * sizeof (GLubyte));

	if (!image->data) {
		fprintf (stderr, "image: malloc failed\n");
		free (image);
		return NULL;
	}
	memcpy (image->data, data, byte_count);

	return image;
}

#if 1
// load png file
struct image_t *
image_load_png (const char *filename) {
	return image_read_png (filename);
}
#else // TODO
// load apng file
struct image_t *
image_load_apng (const char *filename) {
	return image_read_apng (filename);
}
#endif

// load gl texture
struct image_t *
image_load_gl_texture (GLenum target, GLint level) {
	GLint iformat;
	GLenum format, type;

	glGetTexLevelParameteriv (target, level, GL_TEXTURE_INTERNAL_FORMAT, &iformat);
	switch (iformat) {
		case GL_R8:
		case GL_RG8:
		case GL_RGB8:
		case GL_RGBA8:
		case GL_R8_SNORM:
		case GL_RG8_SNORM:
		case GL_RGB8_SNORM:
		case GL_RGBA8_SNORM:
		case GL_R3_G3_B2:
		case GL_RGB4:
		case GL_RGB5:
#if GL_RGB565
		case GL_RGB565:
#endif
		case GL_RGBA2:
		case GL_RGBA4:
		case GL_RGB5_A1:
		case GL_SRGB8:
		case GL_SRGB8_ALPHA8:
		case GL_R8I:
		case GL_RG8I:
		case GL_RGB8I:
		case GL_RGBA8I:
		case GL_R8UI:
		case GL_RG8UI:
		case GL_RGB8UI:
		case GL_RGBA8UI:
			type = GL_UNSIGNED_BYTE;
			break;

		case GL_R16:
		case GL_RG16:
		case GL_RGB16:
		case GL_RGBA16:
		case GL_R16_SNORM:
		case GL_RG16_SNORM:
		case GL_RGB16_SNORM:
		case GL_RGBA16_SNORM:
		case GL_RGB10:
		case GL_RGB12:
		case GL_RGB10_A2:
		case GL_RGB10_A2UI:
		case GL_RGBA12:
		case GL_R16I:
		case GL_RG16I:
		case GL_RGB16I:
		case GL_RGBA16I:
		case GL_R16UI:
		case GL_RG16UI:
		case GL_RGB16UI:
		case GL_RGBA16UI:
			type = GL_UNSIGNED_SHORT;
			break;

		case GL_R32I:
		case GL_RG32I:
		case GL_RGB32I:
		case GL_RGBA32I:
		case GL_R32UI:
		case GL_RG32UI:
		case GL_RGB32UI:
		case GL_RGBA32UI:
			type = GL_UNSIGNED_INT;
			break;

		case GL_R16F:
		case GL_RG16F:
		case GL_RGB16F:
		case GL_RGBA16F:
		case GL_R32F:
		case GL_RG32F:
		case GL_RGB32F:
		case GL_RGBA32F:
		case GL_R11F_G11F_B10F:
		case GL_RGB9_E5:
			type = GL_FLOAT;
			break;

		default:
			fprintf (stderr, "image: internal format %i is unsupported\n", iformat);
			return NULL;
	}

	switch (iformat) {
		case GL_R8:
		case GL_R8_SNORM:
		case GL_R16:
		case GL_R16_SNORM:
		case GL_R16F:
		case GL_R32F:
		case GL_R8I:
		case GL_R8UI:
		case GL_R16I:
		case GL_R16UI:
		case GL_R32I:
		case GL_R32UI:
			format = GL_RED;
			break;

		case GL_RG8:
		case GL_RG8_SNORM:
		case GL_RG16:
		case GL_RG16_SNORM:
		case GL_RG16F:
		case GL_RG32F:
		case GL_RG8I:
		case GL_RG8UI:
		case GL_RG16I:
		case GL_RG16UI:
		case GL_RG32I:
		case GL_RG32UI:
			format = GL_RG;
			break;

		case GL_RGB8:
		case GL_RGB8_SNORM:
		case GL_RGB16:
		case GL_RGB16_SNORM:
		case GL_RGB16F:
		case GL_RGB32F:
		case GL_R3_G3_B2:
		case GL_RGB4:
		case GL_RGB5:
#if GL_RGB565
		case GL_RGB565:
#endif
		case GL_R11F_G11F_B10F:
		case GL_RGB9_E5:
		case GL_SRGB8:
		case GL_RGB8I:
		case GL_RGB8UI:
		case GL_RGB16I:
		case GL_RGB16UI:
		case GL_RGB32I:
		case GL_RGB32UI:
			format = GL_RGB;
			break;

		case GL_RGBA8:
		case GL_RGBA8_SNORM:
		case GL_RGBA16:
		case GL_RGBA16_SNORM:
		case GL_RGBA16F:
		case GL_RGBA32F:
		case GL_RGBA2:
		case GL_RGBA4:
		case GL_RGB5_A1:
		case GL_SRGB8_ALPHA8:
		case GL_RGBA8I:
		case GL_RGBA8UI:
		case GL_RGBA16I:
		case GL_RGBA16UI:
		case GL_RGBA32I:
		case GL_RGBA32UI:
			format = GL_RGBA;
			break;

		default:
			fprintf (stderr, "image: internal format %i is unsupported\n", iformat);
			return NULL;
	}

	return image_load_gl_texture_formatted (target, level, format, type);
}

struct image_t *
image_load_gl_texture_formatted (GLenum target, int level,
                                 GLenum format, GLenum type) {
	struct image_t *image = malloc (sizeof (*image));
	image_gl_pixel_store_state_t state;

	if (!image) {
		fprintf (stderr, "image: malloc failed\n");
		return NULL;
	}

	glGetTexLevelParameteriv (target, level, GL_TEXTURE_WIDTH, &image->width);
	glGetTexLevelParameteriv (target, level, GL_TEXTURE_HEIGHT, &image->height);
	glGetTexLevelParameteriv (target, level, GL_TEXTURE_DEPTH, &image->depth);
	if (image->height == 0) image->height = 1;
	if (image->depth == 0) image->depth = 1;
	image->format = format;
	image->type = type;
	image->data = malloc (sizeof (GLubyte) * image_bytespp (image) 
	            * image->width * image->height * image->depth);

	if (!image->data) {
		fprintf (stderr, "image: malloc failed\n");
		free (image);
		return NULL;
	}

	// get data
	image_get_pixel_pack_state (state);
	image_default_pixel_pack_state ();
	glGetTexImage (target, level, format, type, image->data);
	image_set_pixel_pack_state (state);

	return image;
}

// load gl draw buffer, automatic format
struct image_t *
image_load_gl_colorbuffer (GLenum buffer) {
	GLint rs, gs, bs, as, ct;
	GLenum format, type;

	// retrieve buffer information
	glGetFramebufferAttachmentParameteriv (GL_READ_FRAMEBUFFER, buffer,
		GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &rs);
	glGetFramebufferAttachmentParameteriv (GL_READ_FRAMEBUFFER, buffer,
		GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE, &gs);
	glGetFramebufferAttachmentParameteriv (GL_READ_FRAMEBUFFER, buffer,
		GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &bs);
	glGetFramebufferAttachmentParameteriv (GL_READ_FRAMEBUFFER, buffer,
		GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE, &as);
	glGetFramebufferAttachmentParameteriv (GL_READ_FRAMEBUFFER, buffer,
		GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &ct);

	// check bit values for format
	if (as > 0) format = GL_RGBA;
	else if (bs > 0) format = GL_RGB;
	else if (gs > 0) format = GL_RG;
	else format = GL_RED;

	// choose best type
	if (ct == GL_UNSIGNED_NORMALIZED || ct == GL_SIGNED_NORMALIZED) {
		if (rs <= 8) type = GL_UNSIGNED_BYTE;
		else if (rs <= 16) type = GL_UNSIGNED_SHORT;
		else type = GL_UNSIGNED_INT;
	} // normalized values
	else if (ct == GL_INT || GL_UNSIGNED_INT) {
		if (rs <= 8) type = GL_UNSIGNED_BYTE;
		else if (rs <= 16) type = GL_UNSIGNED_SHORT;
		else type = GL_UNSIGNED_INT;
	} // integer texture
	else if (ct == GL_FLOAT) {
		type = GL_FLOAT;
	} // float value
	else {
		fprintf (stderr, "image: GL component type %i unsupported\n", ct);
		return IMAGE_FAILURE;
	} // unsupported

	return image_load_gl_colorbuffer_formatted (buffer, format, type);
}

// load formatted gl draw buffer
struct image_t *
image_load_gl_colorbuffer_formatted (GLenum buffer, GLenum format, GLenum type) {
	struct image_t *image = malloc (sizeof (*image));
	image_gl_pixel_store_state_t state;
	GLint v[4];

	if (!image) {
		fprintf (stderr, "image: malloc failed\n");
		return NULL;
	}

	// set image dimensions from viewport
	glGetIntegerv(GL_VIEWPORT, v);
	image->width  = v[2];
	image->height = v[3];
	image->depth  = 1;
	image->format = format;
	image->type   = type;
	image->data   = malloc (sizeof (GLubyte) * v[2] * v[3] * image_bytespp (image));
	if (!image->data) {
		fprintf (stderr, "image: malloc failed\n");
		free (image);
		return NULL;
	}

	// get data
	image_get_pixel_pack_state (state);
	image_default_pixel_pack_state ();
	glReadPixels (v[0], v[1], v[2], v[3], format, type, image->data);
	image_set_pixel_pack_state (state);

	return image;
}


//---------------------------------------------------------------------------
// Release image
void image_release (struct image_t *image) {
	free (image->data);
	free (image);
}

//---------------------------------------------------------------------------
// Accessors
int image_get_width (const struct image_t *image) { return image->width; }
int image_get_height (const struct image_t *image) { return image->height; }
int image_get_depth (const struct image_t *image) { return image->depth; }
GLenum image_get_gl_format (const struct image_t *image) { return image->format; }
GLenum image_get_gl_type (const struct image_t *image) { return image->type; }
const void *image_get_data (const struct image_t *image) { return (const void *)image->data; }

//---------------------------------------------------------------------------
// Load OpenGL

int
image_glTexImage1D (GLenum target, GLint level, GLint internalFormat,
                    GLboolean immutable, GLboolean mipmap,
                    const struct image_t *data) {
#if GL_ARB_texture_storage
	if (immutable) {
		GLint levels = image_pot_exp (data->width);

		if (!mipmap) levels = 1;
		glTexStorage1D (target, levels, internalFormat,
		                data->width);
	} // texStorage
	else {
#endif
		glTexImage1D (target, level, internalFormat,
		              data->width,
		              0, data->format, data->type, NULL);
#if GL_ARB_texture_storage
	} // vs texImage
#endif

	if (!image_glTexSubImage1D (target, level, 0, data)) return IMAGE_FAILURE;
	if (mipmap) glGenerateMipmap (target);

	return IMAGE_SUCCESS;
}

int
image_glTexImage2D (GLenum target, GLint level, GLint internalFormat, 
                    GLboolean immutable, GLboolean mipmap,
                    const struct image_t *data) {
#if GL_ARB_texture_storage
	if (immutable) {
		GLint levels = image_pot_exp (IMAGE_MAX2 (data->width, data->height));

		if (!mipmap) levels = 1;
		glTexStorage2D (target, levels, internalFormat,
		                data->width, data->height);
	} // texStorage
	else {
#endif
		glTexImage2D (target, level, internalFormat,
		              data->width, data->height,
		              0, data->format, data->type, NULL);
#if GL_ARB_texture_storage
	} // vs texImage
#endif

	if (!image_glTexSubImage2D (target, level, 0, 0, data)) return IMAGE_FAILURE;
	if (mipmap) glGenerateMipmap (target);

	return IMAGE_SUCCESS;
}

int
image_glTexImage3D (GLenum target, GLint level, GLint internalFormat,
                    GLboolean immutable, GLboolean mipmap,
                    const struct image_t *data) {
#if GL_ARB_texture_storage
	if (immutable) {
		GLint levels = image_pot_exp (IMAGE_MAX3(data->width, data->height, data->depth));

		if (!mipmap) levels = 1;
		glTexStorage3D (target, levels, internalFormat,
		                data->width, data->height, data->depth);
	} // texStorage
	else {
#endif
		glTexImage3D (target, level, internalFormat,
		              data->width, data->height, data->depth,
		              0, data->format, data->type, NULL);
#if GL_ARB_texture_storage
	} // vs texImage
#endif

	if (!image_glTexSubImage3D (target, level, 0, 0, 0, data)) return IMAGE_FAILURE;
	if (mipmap) glGenerateMipmap (target);

	return IMAGE_SUCCESS;
}

int
image_glTexSubImage1D (GLenum target, GLint level,
                       GLint xoffset,
                       const struct image_t *data) {
	image_gl_pixel_store_state_t state;

	image_get_pixel_unpack_state (state);
	image_default_pixel_unpack_state ();
	glTexSubImage1D (target, level, xoffset,
	                 data->width,
	                 data->format, data->type, data->data);
	image_set_pixel_unpack_state (state);

	return IMAGE_SUCCESS;
}

int
image_glTexSubImage2D (GLenum target, GLint level,
                       GLint xoffset, GLint yoffset,
                       const struct image_t *data) {
	image_gl_pixel_store_state_t state;

	image_get_pixel_unpack_state (state);
	image_default_pixel_unpack_state ();
	glTexSubImage2D (target, level, xoffset, yoffset,
	                 data->width, data->height,
	                 data->format, data->type, data->data);
	image_set_pixel_unpack_state (state);

	return IMAGE_SUCCESS;
}

int
image_glTexSubImage3D (GLenum target, GLint level,
                       GLint xoffset, GLint yoffset, GLint zoffset,
                       const struct image_t *data) {
	image_gl_pixel_store_state_t state;

	image_get_pixel_unpack_state (state);
	image_default_pixel_unpack_state ();
	glTexSubImage3D (target, level, xoffset, yoffset, zoffset,
	                 data->width, data->height, data->depth,
	                 data->format, data->type, data->data);
	image_set_pixel_unpack_state (state);

	return IMAGE_SUCCESS;
}

//---------------------------------------------------------------------------
// Exports

int
image_save_gl_front_buffer (void) {
	const int maxframe = 99999;
	const size_t k = 32; // capacity
	static int count = 0;
	GLchar buf[k];

	if (count > maxframe) {
		fprintf (stderr, "image: maximum frame reached\n");
		return IMAGE_FAILURE;
	}

	     if (count < 10 ) snprintf (buf, k, "screenshot%04i.png", count);
	else if (count < 100 ) snprintf (buf, k, "screenshot%03i.png", count);
	else if (count < 1000 ) snprintf (buf, k, "screenshot%02i.png", count);
	else if (count < 10000 ) snprintf (buf, k, "screenshot%01i.png", count);
	else if (count < 100000 ) snprintf (buf, k, "screenshot%i.png", count);

	if (!image_save_gl_front_buffer_as (buf))
		return IMAGE_FAILURE;

	++count;
	return IMAGE_SUCCESS;
}

int
image_save_gl_front_buffer_as (const char *filename) {
	struct image_t *image;
	GLint read_framebuffer, read_buffer;

	// set GL state
	glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &read_framebuffer);
	glGetIntegerv (GL_READ_BUFFER             , &read_buffer);
	glBindFramebuffer (GL_READ_FRAMEBUFFER, 0);
	glReadBuffer (GL_FRONT);

	image = image_load_gl_colorbuffer_formatted (GL_FRONT,
	                                             GL_RGBA,
	                                             GL_UNSIGNED_BYTE);

	// restore GL state
	glBindFramebuffer(GL_READ_FRAMEBUFFER, read_framebuffer);
	glReadBuffer(read_buffer);

	if (!image)
		return IMAGE_FAILURE;

	if (!image_write_png (image, filename)) {
		image_release (image);
		return IMAGE_FAILURE;
	}

	image_release (image);
	return IMAGE_SUCCESS;
}

#if 0 // TODO
int
image_save_apng (const struct image_t *image, const char *filename) {
	if (!image_is_png_ready (image)) {
		fprintf (stderr, "image: image is apng incompatible\n");
		return IMAGE_FAILURE;
	} // apng incompatible

	return image_write_apng (); // TODO
	return IMAGE_SUCCESS;
}
#endif

int
image_save_png (const struct image_t *image, const char *filename) {
	struct image_t *it = malloc (sizeof (*it));
	GLint i, r;

	if (!it) {
		fprintf (stderr, "malloc failed\n");
		return IMAGE_FAILURE;
	}

	memcpy (it, image, sizeof (*image));
	it->depth  = 1;
	for (i = 0, r = IMAGE_SUCCESS; i < image->depth && r == IMAGE_SUCCESS; ++i) {
		bstring b = image->depth > 1 ? bformat ("%s_%i.png", filename, i)
		                             : bformat ("%s.png", filename);

		it->data+= i * image->width * image->height * image_bytespp (image);
		r = image_write_png (it, bstr2cstr (b, '!'));
		bdestroy (b);
	}

	free (it);
	return r;
}


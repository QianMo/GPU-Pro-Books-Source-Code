// ft.c - public domain opengl3.3+ rendering library
// TODO create font utility
// TODO convert data uchar array to uint array (more compact)
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "glload.h"
#include "ft.h"

typedef struct {
	GLuint glbuffer;
	GLuint gltexture;
	GLuint glvertexarray;
	GLuint glprogram;
	GLint gluniform_size_location;
	size_t stream_offset;
} ft_resources_t;

typedef struct {
	unsigned char letter;
	unsigned char xy[3]; // x,y in NDC
} ft_vertex_t; // 4 Bytes

// shader code
static const char *ft_vertex_shader_src[] = {
	"#version 330\n",
	"layout(location=0) in uint iData;\n",
	"void main() {\n",
		"vec2 ndc = vec2(iData >> 8u & 0xFFFu, iData >> 20u & 0xFFFu);\n",
		"gl_Position.xy = ndc/4095.0*2.0-1.0;\n",
		"gl_Position.zw = vec2(iData & 0xFFu,0);\n",
	"}\n"
};
static const char *ft_geometry_shader_src[] = {
	"#version 330\n",
	"layout(points) in;\n",
	"layout(triangle_strip, max_vertices=4) out;\n",
	"uniform vec2 uS;\n", // size of the letter in NDC space
	"out vec2 st;\n",
	"void main() {\n",
		"vec4 d = gl_in[0].gl_Position;\n", // letter aabb
		"st = d.zw * 8.0;\n",
		"gl_Position = vec4(d.xy-vec2(0,uS.y),-1,1);\n",
		"EmitVertex();\n",
		"st.s+= 8;\n",
		"gl_Position = vec4(d.xy+vec2(uS.x,-uS.y),-1,1);\n",
		"EmitVertex();\n",
		"st+= vec2(-8,8);\n",
		"gl_Position = vec4(d.xy,-1,1);\n",
		"EmitVertex();\n",
		"st.s+= 8;\n",
		"gl_Position = vec4(d.xy+vec2(uS.x,0),-1,1);\n",
		"EmitVertex();\n",
		"EndPrimitive();\n"
	"}\n"
};
static const char *ft_fragment_shader_src[] = {
	"#version 330\n",
	"uniform sampler2DRect sFont;\n",
	"in vec2 st;\n",
	"layout(location=0) out vec3 oColour;\n",
	"void main() {\n",
		"oColour = texture(sFont,st).rrr;\n",
		"if(oColour.r == 0.0) discard;\n",
	"}\n"
};

// rle encoded texture data
static const unsigned char ft_rle_texels[] = {
	68, 3, 64, 3, 10, 3, 46, 3, 46, 3, 190, 3, 80, 5, 254, 16, 3, 156, 5, 16, 
	3, 8, 5, 26, 13, 120, 5, 42, 5, 92, 3, 20, 3, 118, 5, 34, 3, 12, 3, 12, 
	3, 32, 5, 254, 254, 32, 3, 44, 7, 10, 3, 4, 3, 8, 7, 2, 3, 24, 3, 14, 
	3, 46, 3, 30, 3, 12, 3, 14, 7, 12, 3, 10, 11, 8, 7, 14, 3, 10, 7, 10, 
	7, 12, 3, 12, 7, 10, 5, 14, 3, 14, 3, 62, 3, 12, 3, 12, 3, 6, 3, 6, 
	9, 10, 7, 8, 7, 12, 9, 8, 3, 14, 7, 8, 3, 6, 3, 10, 3, 12, 7, 8, 
	3, 6, 3, 8, 9, 6, 3, 6, 3, 6, 3, 6, 3, 8, 7, 8, 3, 16, 7, 8, 
	3, 6, 3, 8, 7, 12, 3, 12, 7, 12, 3, 12, 3, 2, 3, 8, 3, 6, 3, 10, 
	3, 12, 9, 10, 3, 18, 3, 10, 3, 42, 9, 12, 7, 8, 7, 12, 5, 12, 7, 10, 
	5, 12, 3, 18, 3, 8, 3, 4, 3, 10, 3, 16, 3, 10, 3, 4, 3, 12, 3, 8, 
	3, 2, 3, 2, 3, 8, 3, 4, 3, 10, 5, 10, 3, 20, 3, 8, 3, 14, 7, 14, 
	3, 12, 5, 12, 3, 12, 3, 2, 3, 8, 3, 6, 3, 10, 3, 12, 9, 10, 3, 14, 
	3, 14, 3, 28, 3, 4, 3, 254, 254, 74, 3, 2, 3, 2, 3, 8, 3, 2, 3, 2, 
	3, 4, 3, 6, 3, 24, 3, 18, 3, 28, 3, 14, 3, 46, 3, 10, 3, 6, 3, 10, 
	3, 12, 3, 12, 3, 6, 3, 12, 3, 8, 3, 6, 3, 6, 3, 6, 3, 10, 3, 10, 
	3, 6, 3, 12, 3, 28, 3, 16, 3, 26, 3, 28, 3, 2, 7, 6, 3, 6, 3, 6, 
	3, 6, 3, 6, 3, 6, 3, 6, 3, 4, 3, 10, 3, 14, 3, 12, 3, 6, 3, 6, 
	3, 6, 3, 10, 3, 10, 3, 6, 3, 6, 3, 4, 3, 10, 3, 12, 3, 6, 3, 6, 
	3, 4, 5, 6, 3, 6, 3, 6, 3, 14, 3, 2, 3, 2, 3, 6, 3, 4, 3, 8, 
	3, 6, 3, 10, 3, 10, 3, 6, 3, 10, 3, 12, 3, 2, 3, 8, 3, 6, 3, 10, 
	3, 12, 3, 16, 3, 16, 3, 12, 3, 44, 3, 4, 3, 8, 3, 4, 3, 8, 3, 4, 
	3, 8, 3, 4, 3, 8, 3, 4, 3, 8, 3, 16, 3, 14, 7, 8, 3, 4, 3, 10, 
	3, 16, 3, 10, 3, 2, 3, 12, 3, 10, 3, 2, 3, 2, 3, 8, 3, 4, 3, 8, 
	3, 4, 3, 8, 7, 12, 7, 8, 3, 20, 3, 10, 3, 12, 3, 4, 3, 10, 3, 12, 
	3, 2, 3, 10, 3, 2, 3, 12, 3, 12, 3, 16, 3, 14, 3, 14, 3, 26, 3, 2, 
	5, 2, 3, 254, 254, 28, 3, 26, 3, 2, 3, 14, 3, 2, 3, 10, 5, 2, 3, 4, 
	3, 6, 3, 24, 3, 18, 3, 8, 3, 6, 3, 10, 3, 62, 3, 10, 5, 4, 3, 10, 
	3, 14, 3, 18, 3, 6, 11, 14, 3, 6, 3, 6, 3, 10, 3, 10, 3, 6, 3, 14, 
	3, 42, 3, 10, 11, 10, 3, 14, 3, 10, 5, 2, 3, 2, 3, 4, 11, 6, 3, 6, 
	3, 6, 3, 14, 3, 6, 3, 8, 3, 14, 3, 12, 3, 6, 3, 6, 3, 6, 3, 10, 
	3, 10, 3, 6, 3, 6, 3, 2, 3, 12, 3, 12, 3, 6, 3, 6, 3, 4, 5, 6, 
	3, 6, 3, 6, 3, 14, 3, 6, 3, 6, 3, 2, 3, 18, 3, 10, 3, 10, 3, 6, 
	3, 8, 3, 2, 3, 10, 3, 2, 3, 10, 3, 2, 3, 12, 3, 12, 3, 16, 3, 16, 
	3, 12, 3, 44, 3, 16, 7, 8, 3, 4, 3, 8, 3, 14, 3, 4, 3, 8, 9, 10, 
	3, 12, 3, 4, 3, 8, 3, 4, 3, 10, 3, 16, 3, 10, 5, 14, 3, 10, 3, 2, 
	3, 2, 3, 8, 3, 4, 3, 8, 3, 4, 3, 8, 3, 4, 3, 8, 3, 4, 3, 8, 
	3, 16, 5, 12, 3, 12, 3, 4, 3, 8, 3, 2, 3, 8, 3, 2, 3, 2, 3, 10, 
	3, 12, 3, 2, 3, 12, 3, 14, 3, 30, 3, 26, 5, 6, 3, 254, 254, 28, 3, 26, 
	11, 8, 7, 10, 9, 8, 3, 2, 3, 2, 3, 22, 3, 18, 3, 10, 3, 2, 3, 8, 
	11, 24, 9, 28, 3, 8, 3, 2, 3, 2, 3, 10, 3, 16, 3, 10, 7, 8, 3, 4, 
	3, 16, 3, 6, 9, 14, 3, 10, 7, 10, 9, 40, 3, 34, 3, 14, 3, 8, 5, 2, 
	3, 2, 3, 6, 3, 2, 3, 8, 9, 8, 3, 14, 3, 6, 3, 8, 7, 10, 7, 8, 
	3, 4, 5, 6, 11, 10, 3, 18, 3, 6, 5, 14, 3, 12, 3, 6, 3, 6, 3, 2, 
	3, 2, 3, 6, 3, 6, 3, 6, 9, 8, 3, 6, 3, 6, 9, 10, 7, 12, 3, 10, 
	3, 6, 3, 8, 3, 2, 3, 8, 3, 2, 3, 2, 3, 10, 3, 14, 3, 14, 3, 14, 
	3, 14, 3, 14, 3, 42, 7, 18, 3, 8, 3, 4, 3, 8, 3, 4, 3, 8, 3, 4, 
	3, 8, 3, 4, 3, 10, 3, 12, 3, 4, 3, 8, 3, 4, 3, 10, 3, 16, 3, 10, 
	3, 2, 3, 12, 3, 10, 3, 2, 3, 2, 3, 8, 3, 4, 3, 8, 3, 4, 3, 8, 
	3, 4, 3, 8, 3, 4, 3, 8, 5, 12, 3, 16, 3, 12, 3, 4, 3, 8, 3, 2, 
	3, 8, 3, 6, 3, 8, 3, 2, 3, 10, 3, 2, 3, 14, 3, 10, 3, 34, 3, 24, 
	5, 6, 3, 254, 254, 28, 3, 28, 3, 2, 3, 8, 3, 2, 3, 10, 3, 2, 5, 12, 
	3, 28, 3, 18, 3, 10, 7, 12, 3, 64, 3, 8, 3, 4, 5, 10, 3, 18, 3, 10, 
	3, 12, 3, 2, 3, 8, 9, 8, 3, 20, 3, 8, 3, 6, 3, 6, 3, 6, 3, 10, 
	3, 14, 3, 14, 3, 10, 11, 10, 3, 18, 3, 6, 3, 2, 5, 2, 3, 6, 3, 2, 
	3, 8, 3, 6, 3, 6, 3, 14, 3, 6, 3, 8, 3, 14, 3, 12, 3, 14, 3, 6, 
	3, 10, 3, 18, 3, 6, 3, 2, 3, 12, 3, 12, 3, 2, 3, 2, 3, 6, 3, 2, 
	3, 2, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 
	3, 18, 3, 10, 3, 6, 3, 6, 3, 6, 3, 6, 3, 2, 3, 2, 3, 8, 3, 2, 
	3, 10, 3, 2, 3, 14, 3, 12, 3, 14, 3, 14, 3, 44, 3, 16, 5, 10, 7, 12, 
	5, 12, 7, 10, 5, 10, 7, 12, 7, 8, 7, 12, 3, 16, 3, 10, 3, 4, 3, 10, 
	3, 10, 5, 2, 3, 10, 7, 12, 5, 10, 7, 12, 7, 8, 3, 2, 5, 10, 7, 8, 
	7, 10, 3, 4, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 
	3, 8, 9, 10, 3, 14, 3, 14, 3, 26, 3, 2, 5, 2, 3, 254, 254, 28, 3, 12, 
	3, 2, 3, 8, 11, 6, 3, 2, 3, 2, 3, 6, 3, 2, 3, 2, 3, 8, 3, 2, 
	3, 12, 3, 14, 3, 14, 3, 10, 3, 2, 3, 2, 3, 10, 3, 66, 3, 6, 3, 6, 
	3, 8, 5, 10, 3, 6, 3, 12, 3, 12, 5, 8, 3, 16, 3, 20, 3, 6, 3, 6, 
	3, 6, 3, 6, 3, 44, 3, 26, 3, 14, 3, 4, 3, 8, 3, 4, 3, 10, 3, 10, 
	3, 6, 3, 6, 3, 6, 3, 6, 3, 4, 3, 10, 3, 14, 3, 12, 3, 6, 3, 6, 
	3, 6, 3, 10, 3, 18, 3, 6, 3, 4, 3, 10, 3, 12, 5, 2, 5, 6, 5, 4, 
	3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 
	3, 10, 3, 10, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 
	3, 6, 3, 14, 3, 10, 3, 12, 3, 16, 3, 12, 3, 2, 3, 26, 3, 30, 3, 36, 
	3, 26, 3, 28, 3, 46, 3, 16, 3, 126, 3, 110, 3, 14, 3, 14, 3, 10, 3, 2, 
	5, 10, 3, 4, 3, 254, 254, 30, 3, 12, 3, 2, 3, 12, 3, 2, 3, 8, 7, 10, 
	9, 10, 3, 14, 3, 16, 3, 10, 3, 16, 3, 82, 3, 8, 7, 12, 3, 12, 7, 8, 
	11, 12, 3, 8, 11, 10, 5, 8, 11, 8, 7, 10, 7, 92, 5, 12, 5, 12, 3, 10, 
	9, 10, 7, 8, 7, 12, 9, 8, 9, 8, 7, 8, 3, 6, 3, 10, 3, 18, 3, 6, 
	3, 6, 3, 8, 3, 12, 3, 6, 3, 6, 5, 4, 3, 8, 7, 8, 9, 10, 7, 8, 
	9, 10, 7, 8, 11, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 
	3, 6, 3, 6, 3, 8, 9, 10, 5, 10, 3, 14, 5, 14, 3, 30, 5, 26, 3, 36, 
	3, 28, 5, 24, 3, 16, 3, 16, 3, 10, 3, 16, 3, 126, 3, 112, 3, 12, 3, 12, 
	3, 14, 5, 2, 3, 10, 5, 254, 254, 12
};

// library resources
static ft_resources_t *ft_resources = NULL;

// constants
const size_t ft_texture_width  = 1024; // font sheet width
const size_t ft_texture_height = 8; // font sheet height
const size_t ft_buffer_byte_size = 1<<23; // 8 MB stream buffer
const size_t ft_max_stream_count = 1024;  // maximum streamed vertices per drawcall
const size_t ft_letter_pixel_width = 7; // char space
const size_t ft_tab_width = 1<<4; // tab width

// macros 
#define FT_MIN(a,b) ((a) > (b) ? (b) : (a))

/*
find next power of two (having x>0)
*/
static int
ft_pot(int x) {
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x++;
	return x;
}

/*
unpack rle texture to memory
*/
static int
ft_unpack_texels(unsigned char *dst) {
	unsigned char *data = dst;
	size_t count = sizeof(ft_rle_texels);
	size_t i, rle_count;
	GLint colour = 0;

	for(i = 0; i < count && i < ft_texture_width * ft_texture_height; ++i) {
		rle_count = (ft_rle_texels[i] >> 1u) & 0x7Fu;
		colour = (ft_rle_texels[i] & 1) * 255;
		memset(data, colour, rle_count);
		data+= rle_count;
	}
#ifndef NDEBUG
	// should never fail
	if(data - dst != ft_texture_width * ft_texture_height) {
		fprintf(stderr,"ft_debug: failed to unpack texel data\n");
		return FT_FAILURE;
	}
#endif
	return FT_SUCCESS;
}

/*
create shader
*/
static int
ft_create_shader(GLuint shader, const char** src, size_t n) {
	glShaderSource(shader, n, src, NULL);
	glCompileShader(shader);
#ifndef NDEBUG 
	// should never fail
	{
		int is_compiled = 0;

		glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
		if(!is_compiled) {
			char buffer[256];

			glGetShaderInfoLog(shader, 256, NULL, buffer);
			fprintf(stderr, "ft_error: GLSL compilation " \
			                "error:\n %s\n", buffer);
			return FT_FAILURE;
		}
	}
#endif
	return FT_SUCCESS;
}

/* 
build vertex data from a letter
*/
static void
ft_stream_letter(char letter,
                 ft_vertex_t* vertex, int x, int y) {
	int x0 = FT_MIN(x, 0xFFF); // clamp to 4095
	int y0 = FT_MIN(y, 0xFFF); // clamp to 4095
	vertex->letter = letter - ' ';   // llll llll ---- ---- ---- ---- ---- ----
	vertex->xy[0] = x0 & 0xFFu;      // llll llll xxxx xxxx ---- ---- ---- ----
	vertex->xy[1] = x0 >> 8 & 0x0Fu; // llll llll xxxx xxxx xxxx ---- ---- ----
	vertex->xy[1]|= y0 << 4 & 0xF0u; // llll llll xxxx xxxx xxxx yyyy ---- ----
	vertex->xy[2] = y0 >> 4 & 0xFFu; // llll llll xxxx xxxx xxxx yyyy yyyy yyyy
}

/*
fill buffer with letters
*/
static int
ft_stream_letters(ft_vertex_t *data,
                  const char *buffer,
                  int font_size,
                  int glviewport_width, int glviewport_height, int y) {
	int count = 0;
	int yc = glviewport_height - y; // invert coord system
	int xc = 0;
	int i = 0;

	for(; buffer[i] != '\0' && yc > 0; ++i) {
		if(buffer[i]=='\n') {
			xc = 0;
			yc-= font_size*2;
		} else if(buffer[i]=='\t') {
			xc+= (ft_tab_width - (xc % ft_tab_width)) 
			   * (font_size >> 3);
		} else if(buffer[i]==' ') {
			xc+= ft_letter_pixel_width * (font_size >> 3);
		} else if(xc < glviewport_width - font_size) {
			ft_stream_letter(buffer[i], data, xc, yc);
			xc+= ft_letter_pixel_width * (font_size >> 3);
			++count;
			++data;
		}
	}
	return count;
}

/*
load buffer object
*/
static int
ft_load_buffer(GLuint buffer) {
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER,
	             ft_buffer_byte_size,
	             NULL,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return FT_SUCCESS;
}

/*
load the vertex array
*/
static int
ft_load_vertex_array(GLuint vertex_array) {
	glBindVertexArray(vertex_array);
	glBindBuffer(GL_ARRAY_BUFFER, ft_resources->glbuffer);
	glEnableVertexAttribArray(0);
	glVertexAttribIPointer(0, 1, GL_UNSIGNED_INT, 0, 0);
	glBindVertexArray(0);
	return FT_SUCCESS;
}

/*
load texture
*/
static int
ft_load_texture(GLuint texture, int gl_texture_unit) {
	size_t texel_count = ft_texture_width * ft_texture_height;
	GLubyte *texels = (GLubyte *) malloc(texel_count);
	GLint max_combined_texture_image_units = 0;

	// check texture unit support
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, 
	              &max_combined_texture_image_units);
	if(gl_texture_unit - GL_TEXTURE0 >= max_combined_texture_image_units) {
		fprintf(stderr, "ft_error: texture unit is greater than " \
		                "platform's GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS\n");
		return FT_FAILURE;
	}
	if(!texels) {
		fprintf(stderr, "ft_error: failed to allocate memory " \
		                "for texel data\n");
		return FT_FAILURE;
	}
	// unpack texels and load texture
	if(!ft_unpack_texels(texels)) {
		free(texels);
		return FT_FAILURE;
	}
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glPixelStorei(GL_UNPACK_SWAP_BYTES, 0);
	glPixelStorei(GL_UNPACK_LSB_FIRST, 0);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
	glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
	glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
	glPixelStorei(GL_UNPACK_SKIP_IMAGES, 0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glActiveTexture(gl_texture_unit);
	glBindTexture(GL_TEXTURE_RECTANGLE, texture);
#if 0 // disabled for compatibility
	glTexStorage2D(GL_TEXTURE_RECTANGLE, 1, GL_R8,
	               ft_texture_width,
	               ft_texture_height);
	glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0,
	                ft_texture_width, ft_texture_height,
	                GL_RED,GL_UNSIGNED_BYTE, texels);
#else
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R8,
	             ft_texture_width, ft_texture_height,
	             0, GL_RED, GL_UNSIGNED_BYTE, texels);
#endif
	glTexParameteri(GL_TEXTURE_RECTANGLE,
	                GL_TEXTURE_MAG_FILTER,
	                GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE,
	                GL_TEXTURE_MIN_FILTER,
	                GL_NEAREST);
	free(texels);
	return FT_SUCCESS;
}

/*
load program
*/
static int
ft_load_program(GLuint program, int gl_texture_unit) {
	size_t vn = sizeof(ft_vertex_shader_src) / sizeof(char *);
	size_t gn = sizeof(ft_geometry_shader_src) / sizeof(char *);
	size_t fn = sizeof(ft_fragment_shader_src) / sizeof(char *);
	GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
	GLuint geometry = glCreateShader(GL_GEOMETRY_SHADER);
	GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
	GLint active_program = 0;

	ft_create_shader(vertex, ft_vertex_shader_src, vn);
	ft_create_shader(geometry, ft_geometry_shader_src, gn);
	ft_create_shader(fragment, ft_fragment_shader_src, fn);
	glAttachShader(program, vertex);
	glAttachShader(program, geometry);
	glAttachShader(program, fragment);
	glDeleteShader(vertex);
	glDeleteShader(geometry);
	glDeleteShader(fragment);
	glLinkProgram(program);
#ifdef NDEBUG
	// should never fail
	{
		GLint link_status = 0;
		glGetProgramiv(program, GL_LINK_STATUS, &link_status);
		if(!link_status) {
			GLchar buffer[256];
			glGetProgramInfoLog(program, 256, NULL, buffer);
			fprintf(stderr, "ft_error: GLSL link error:\n"
			                "%s\n",
			                buffer);
			return FT_FAILURE;
		}
	}
#endif
	glGetIntegerv(GL_CURRENT_PROGRAM, &active_program);
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, "sFont"), 
	            gl_texture_unit - GL_TEXTURE0);
	ft_resources->gluniform_size_location = 
		glGetUniformLocation(program, "uS");
	glUseProgram(active_program);
	return FT_SUCCESS;
}

/*
stream vertices asynchronously
*/
static int
ft_draw_buffer(int font_size, int x, int y, const char *buffer) {
	ft_vertex_t *vertices = NULL;
	GLint glviewport[4] = {0, 0, 0, 0};
	GLint count = 0;

	// get viewport and check coords
	glGetIntegerv(GL_VIEWPORT, glviewport);
	if(x + font_size > glviewport[2] || y + font_size > glviewport[3])
		return FT_SUCCESS; // text is out of viewport

	// map buffer
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, ft_resources->glbuffer);
	vertices = (ft_vertex_t *)
	glMapBufferRange(GL_ARRAY_BUFFER,
	                 sizeof(ft_vertex_t) * ft_resources->stream_offset,
	                 sizeof(ft_vertex_t) * ft_max_stream_count,
	                 GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
	// check memory
	if(!vertices) {
		fprintf(stderr, "ft_error: glMapBufferRange() failed\n");
		return FT_FAILURE;
	}
	// set data
	count = ft_stream_letters(vertices, buffer, font_size,
		                      glviewport[2], glviewport[3], y);
	// unmap buffer
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// draw if data has been generated
	if(count > 0) {
		glViewport(x, 0, 4096, 4096);
		glUseProgram(ft_resources->glprogram);
		glUniform2f(ft_resources->gluniform_size_location,
		            2.f * font_size / 4096.f,
		            2.f * font_size / 4096.f);
		glBindVertexArray(ft_resources->glvertexarray);
		glDrawArrays(GL_POINTS, ft_resources->stream_offset, count);
		glViewport(glviewport[0], glviewport[1],
		           glviewport[2], glviewport[3]);

		// compute new buffer_offset
		ft_resources->stream_offset+= ft_pot(count);
		if((ft_resources->stream_offset + ft_max_stream_count)
		     * sizeof(ft_vertex_t) >= ft_buffer_byte_size) {
			ft_resources->stream_offset = 0; // loop
#ifndef NDEBUG
		fprintf(stderr, "ft_debug: reached buffer capacity, " \
		                "offset reset\n");
#endif
		}
	}
	return FT_SUCCESS;
}

/*
init
*/
int
ft_init(int gl_texture_unit) {
	if(ft_resources) {
		fprintf(stderr, "ft_error: ft has already been called\n");
		return FT_FAILURE;
	}
	ft_resources = (ft_resources_t *) malloc(sizeof(ft_resources_t));
	if(!ft_resources) {
		fprintf(stderr, "ft_error: failed to allocate memory " \
		                "for resources\n");
		return FT_FAILURE;
	}
	// set resources
	ft_resources->stream_offset = 0;
	glGenBuffers(1, &ft_resources->glbuffer);
	if(!ft_load_buffer(ft_resources->glbuffer)) {
		glDeleteBuffers(1, &ft_resources->glbuffer);
		free(ft_resources);
		ft_resources = NULL;
		fprintf(stderr, "ft_error: failed to load GL buffer\n");
		return FT_FAILURE;
	}

	glGenTextures(1, &ft_resources->gltexture);
	if(!ft_load_texture(ft_resources->gltexture, gl_texture_unit)) {
		glDeleteBuffers(1, &ft_resources->glbuffer);
		glDeleteTextures(1, &ft_resources->gltexture);
		free(ft_resources);
		ft_resources = NULL;
		fprintf(stderr, "ft_error: failed to load GL texture\n");
		return FT_FAILURE;
	}

	glGenVertexArrays(1, &ft_resources->glvertexarray);
	if(!ft_load_vertex_array(ft_resources->glvertexarray)) {
		glDeleteVertexArrays(1, &ft_resources->glvertexarray);
		glDeleteBuffers(1, &ft_resources->glbuffer);
		glDeleteTextures(1, &ft_resources->gltexture);
		free(ft_resources);
		ft_resources = NULL;
		fprintf(stderr, "ft_error: failed to load GL vertex array\n");
		return FT_FAILURE;
	}

	ft_resources->glprogram = glCreateProgram();
	if(!ft_load_program(ft_resources->glprogram, gl_texture_unit)) {
		glDeleteProgram(ft_resources->glprogram);
		glDeleteVertexArrays(1, &ft_resources->glvertexarray);
		glDeleteBuffers(1, &ft_resources->glbuffer);
		glDeleteTextures(1, &ft_resources->gltexture);
		free(ft_resources);
		ft_resources = NULL;
		fprintf(stderr, "ft_error: failed to load GL program\n");
		return FT_FAILURE;
	}
	return FT_SUCCESS;
}

/*
shutdown
*/
int
ft_shutdown() {
	// free resources
	if(!ft_resources) {
		fprintf(stderr, "ft_error: ft_shutdown() failed because " \
		                "internal resources are not ready (did " \
		                "you forget to call ft_init() ?)\n");
		return FT_FAILURE;
	}
	glDeleteTextures(1, &ft_resources->gltexture);
	glDeleteBuffers(1, &ft_resources->glbuffer);
	glDeleteVertexArrays(1, &ft_resources->gltexture);
	glDeleteProgram(ft_resources->glprogram);
	free(ft_resources);
	ft_resources = NULL;
	return FT_SUCCESS;
}

/*
print
*/
int
ft_print(int font_size, int x, int y, const char *format, ...) {
	// check format string
	if(!format)
		return FT_SUCCESS;
	// check state
	if(!ft_resources) {
		fprintf(stderr, "ft_error: ft_print() failed because " \
		                "internal resources are not ready (did " \
		                "you forget to call ft_init() ?)\n");
		return FT_FAILURE;
	}
	// check font size
	if(font_size != FT_FONT_SIZE_SMALL &&
	   font_size != FT_FONT_SIZE_MEDIUM &&
	   font_size != FT_FONT_SIZE_LARGE) {
		fprintf(stderr, "ft_error: ft_print() failed because " \
		                "font size %i is unsupported\n",
		                font_size);
		return FT_FAILURE;
	}
	// check coords
	if(x < 0 || y < 0) {
		fprintf(stderr, "ft_error: ft_print() failed because " \
		                "coordinates (%i,%i) are invalid", x, y);
		return FT_FAILURE;
	}
	// render
	if(format[0]!='\0') {
		char buffer[ft_max_stream_count];
		va_list vl;

		// parse string
		va_start(vl, format);
		vsnprintf(buffer, ft_max_stream_count, format, vl);
		va_end(vl);

		// stream vertices
		return ft_draw_buffer(font_size, x, y, buffer);
	}
	return FT_SUCCESS;
}


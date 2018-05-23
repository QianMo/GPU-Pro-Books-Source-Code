#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "bstrlib.h"
#include "buf.h"
#include "glload.h"
#include "program.h"

// log
#define PROGRAM_LOG_SIZE 1024
static char g_program_log[PROGRAM_LOG_SIZE];

// shader stage bits
enum {
	program_stage_error_bit           = -1,
	program_stage_vertex_bit          = 1,
	program_stage_fragment_bit        = 1<<1,
	program_stage_geometry_bit        = 1<<2,
	program_stage_tess_control_bit    = 1<<3,
	program_stage_tess_evaluation_bit = 1<<4,
	program_stage_compute_bit         = 1<<5,
	program_stage_all_bit             = 0x3F
};

/*
convert opengl stage to program stage bit
*/
static int
program_glstage_to_stage_bit(int glstage) {
	switch(glstage) {
		case GL_VERTEX_SHADER:   return program_stage_vertex_bit;
		case GL_FRAGMENT_SHADER: return program_stage_fragment_bit;
#if defined(GL_GEOMETRY_SHADER)
		case GL_GEOMETRY_SHADER: return program_stage_geometry_bit;
#endif
#if defined(GL_TESS_CONTROL_SHADER)
		case GL_TESS_CONTROL_SHADER: return program_stage_tess_control_bit;
#endif
#if defined(GL_TESS_EVALUATION_SHADER)
		case GL_TESS_EVALUATION_SHADER: return program_stage_tess_evaluation_bit;
#endif
#if defined(GL_COMPUTE_SHADER)
		case GL_COMPUTE_SHADER: return program_stage_compute_bit;
#endif
		default:
			fprintf(stderr, "program: unknown shader stage\n");
			return program_stage_error_bit;
	};
}

/*
loads, compiles and attaches a shader to a program
*/
static int
program_attach_shader(GLuint program,
                      GLenum shader_type,
                      const GLchar **source,
                      GLsizei count) {
	GLint is_compiled = 0;
	GLuint shader = glCreateShader(shader_type);

	// set source and compile
	glShaderSource(shader, count, source, NULL);
	glCompileShader(shader);

	// check compilation
	glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
	if(is_compiled == GL_FALSE) {
#if 0
		GLint log_length = 0;
		GLchar *log = NULL;

		// get log data
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
		log = (GLchar *) malloc(log_length);
		glGetShaderInfoLog(shader, log_length, NULL, log_buf);
		free(log);
#else
		glGetShaderInfoLog(shader, PROGRAM_LOG_SIZE, NULL, g_program_log);
#endif
		glDeleteShader(shader);
		return PROGRAM_ERR;
	}

	// attach shader and flag for deletion
	glAttachShader(program, shader);
	glDeleteShader(shader);
	return PROGRAM_OK;
}


/*
*/
struct program_args_t {
	char **files;
	bstring header;
	int version;
	int is_compatible;
	int stage_bits;
};


/*
create
*/
struct program_args_t *
program_args_create(void) {
	struct program_args_t *args = malloc(sizeof(*args));

	if(!args) {
		fprintf(stderr, "program: malloc failed\n");
	}
	args->files = NULL;
	args->header  = bfromcstr("");
	args->version = 0;
	args->is_compatible = false;
	args->stage_bits = program_stage_all_bit;
	return args;
}

/*
release
*/
void
program_args_release(struct program_args_t *args) {
	int i = 0;

	for(; i<buf_len(args->files); ++i)
		bcstrfree(args->files[i]);
	buf_free(args->files);
	bdestroy(args->header);
	free(args);
}

/*
push file
*/
int
program_args_push_file(struct program_args_t *args, const char *file) {
	bstring str = bfromcstr(file);

	buf_push(args->files, bstr2cstr(str, ' '));
	bdestroy(str);
	return PROGRAM_OK;
}

/*
push head
*/
int
program_args_push_string(struct program_args_t *args, const char *str, ...) {
	static const size_t buf_len = 64;
	char buf[buf_len];
	va_list vl;
	int n = 0;

	va_start(vl, str);
	n = vsnprintf(buf, buf_len, str, vl);
	va_end(vl);
	if(n < 0 || n >= buf_len) {
		perror("program: vsnprintf failed: ");
		return PROGRAM_ERR;
	}
	bcatcstr(args->header, buf);
	bcatcstr(args->header, "\n");
	return PROGRAM_OK;
}

/*
set version
*/
void
program_args_set_version(struct program_args_t *args, unsigned int version, bool compatible) {
	args->version = version;
	args->is_compatible = compatible;
}

/*
allow / forbid stage
*/
int
program_args_allow_stage(struct program_args_t *args, int glstage) {
	int stage_bit = program_glstage_to_stage_bit(glstage);

	if(stage_bit == program_stage_error_bit)
		return PROGRAM_ERR;
	args->stage_bits |= stage_bit;
	return PROGRAM_OK;
}

int
program_args_forbid_stage(struct program_args_t *args, int glstage) {
	int stage_bit = program_glstage_to_stage_bit(glstage);

	if(stage_bit == program_stage_error_bit)
		return PROGRAM_ERR;
	args->stage_bits &= ~stage_bit;
	return PROGRAM_OK;
}

/*
create program
*/
GLuint
program_create(const struct program_args_t *args, bool link) {
	GLuint program = 0u;
	GLchar **source_code = NULL;
	GLchar **source_it;
	GLchar *stage_str= malloc(64);

	if(!stage_str) {
		fprintf(stderr, "program: malloc failed\n");
		return 0;
	}

	// check params
	if(args->files == NULL) {
		fprintf(stderr, "program: no files to load\n");
		return 0;
	}

	// generate version
	if(args->version > 0) {
		bstring version_str = 
			bformat("#version %i %s\n",
			        args->version,
			        args->is_compatible ? "compatibility" : "");
		buf_push(source_code, bstr2cstr(version_str, ' '));
		bdestroy(version_str);
	} else { // default version
		bstring version_str = bfromcstr("/* default version */\n");
		buf_push(source_code, bstr2cstr(version_str, ' '));
		bdestroy(version_str);
	}

	// add forbidden stage macros
	if(args->stage_bits != program_stage_all_bit) {
		bstring forbidden_str = bfromcstr("#define SHADER_FORBIDDEN 1\n");
		if(!(args->stage_bits & program_stage_vertex_bit))
			bcatcstr(forbidden_str, "#define VERTEX_SHADER_FORBIDDEN 1\n");
		if(!(args->stage_bits & program_stage_fragment_bit))
			bcatcstr(forbidden_str, "#define FRAGMENT_SHADER_FORBIDDEN 1\n");
		if(!(args->stage_bits & program_stage_geometry_bit))
			bcatcstr(forbidden_str, "#define GEOMETRY_SHADER_FORBIDDEN 1\n");
		if(!(args->stage_bits & program_stage_tess_control_bit))
			bcatcstr(forbidden_str, "#define TESS_CONTROL_SHADER_FORBIDDEN 1\n");
		if(!(args->stage_bits & program_stage_tess_evaluation_bit))
			bcatcstr(forbidden_str, "#define TESS_EVALUATION_SHADER_FORBIDDEN 1\n");
		if(!(args->stage_bits & program_stage_compute_bit))
			bcatcstr(forbidden_str, "#define COMPUTE_SHADER_FORBIDDEN 1\n");
		buf_push(source_code, bstr2cstr(forbidden_str, ' '));
		bdestroy(forbidden_str);
	}

	// reserve shader definition
	buf_push(source_code, stage_str);

	// push options
	if(blength(args->header) > 1 || args->stage_bits != program_stage_all_bit) {
		buf_push(source_code, bstr2cstr(args->header, ' '));
	}

	// push file contents
	buf_foreach(source_it, args->files) {
		FILE *file = fopen(*source_it, "r");
		bstring file_content;

		// read file content
		if(!file) {
			fprintf(stderr, "program: fopen failed: ");
			perror(*source_it);
			return 0;
		}
		file_content = bread((bNread) fread, file);
		fclose(file);

		// push to source
		buf_push(source_code, bstr2cstr(file_content, ' '));
		bdestroy(file_content);
	}

	// create program and attach shaders
	program = glCreateProgram();
	source_it = buf_back(source_code);

#define ATTACH_SHADER(glstage, str, bit)                        \
	if(strstr(*source_it, (str)) && (args->stage_bits & bit)) {  \
		memset(stage_str, ' ', 64);                               \
		snprintf(stage_str, 64, "#define %s 1\n", str);           \
		int c_ = program_attach_shader(program,                    \
		                               glstage,                    \
		                               (const char **)source_code, \
		                               buf_len(source_code));      \
		if(c_ == PROGRAM_ERR) {                                    \
			glDeleteProgram(program);                             \
			buf_foreach(source_it, source_code)                   \
				bcstrfree(*source_it);                            \
			buf_free(source_code);                                \
			return 0;                                             \
		}                                                         \
	}

	ATTACH_SHADER(GL_VERTEX_SHADER, "VERTEX_SHADER", program_stage_vertex_bit);
	ATTACH_SHADER(GL_FRAGMENT_SHADER, "FRAGMENT_SHADER", program_stage_fragment_bit);
#if defined(GL_GEOMETRY_SHADER)
	ATTACH_SHADER(GL_GEOMETRY_SHADER, "GEOMETRY_SHADER", program_stage_geometry_bit);
#endif
#if defined(GL_TESS_CONTROL_SHADER)
	ATTACH_SHADER(GL_TESS_CONTROL_SHADER, "TESS_CONTROL_SHADER", program_stage_tess_control_bit);
#endif
#if defined(GL_TESS_EVALUATION_SHADER)
	ATTACH_SHADER(GL_TESS_EVALUATION_SHADER, "TESS_EVALUATION_SHADER", program_stage_tess_evaluation_bit);
#endif
#if defined(GL_COMPUTE_SHADER)
	ATTACH_SHADER(GL_COMPUTE_SHADER, "COMPUTE_SHADER", program_stage_compute_bit);
#endif

#undef ATTACH_SHADER

	// link if requested
	if(link) {
		GLint link_status = 0;
		glLinkProgram(program);
		glGetProgramiv(program, GL_LINK_STATUS, &link_status);
		if(GL_FALSE == link_status) {
#if 0
			GLint log_length = 0;
			GLchar *log = NULL;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
			log = malloc(log_length);
			glGetProgramInfoLog(program, log_length, NULL, log);
			fprintf(stderr, "program: link error\n",
			        log);
			free(log);
			for(i=1; i<count; ++i)
				free(source_code[i]);
			free(source_code);
#else 
			fprintf(stderr, "program: link failed\n");
			glGetProgramInfoLog(program, PROGRAM_LOG_SIZE, NULL, g_program_log);
#endif
			glDeleteProgram(program);
			return 0;
		}
	}
	// cleanup
	buf_foreach(source_it, source_code)
		bcstrfree(*source_it);
	buf_free(source_code);

	return program;
}

/*
get shader log
*/
const char *
program_get_log(void) {
	return g_program_log;
}


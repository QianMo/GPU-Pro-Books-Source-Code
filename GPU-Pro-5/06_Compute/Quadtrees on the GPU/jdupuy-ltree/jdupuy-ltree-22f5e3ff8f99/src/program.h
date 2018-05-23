#ifndef PROGRAM_H
#define PROGRAM_H

// source stack api
struct program_args_t *program_args_create(void);
void program_args_release(struct program_args_t *args);

#define PROGRAM_OK (0)
#define PROGRAM_ERR (-1)
int program_args_push_string(struct program_args_t *args, const char *string, ...);
int program_args_push_file(struct program_args_t *args, const char *file);
int program_args_forbid_stage(struct program_args_t *args, int glstage);
int program_args_allow_stage(struct program_args_t *args, int glstage);
void program_args_set_version(struct program_args_t *args, unsigned int glslversion, bool compatible);

// create program from source stack
unsigned int program_create(const struct program_args_t *args, bool link);
const char *program_get_log(void);

/*
usage
	struct program_args_t *args = program_args_create();
	program_args_set_version(args, 420, false);
	program_args_push_string(args, "%s", g_test ? "#define test\n" : "");
	program_args_push_file(args, "noise.glsl");
	program_args_push_file(args, "main.glsl");
	GLuint program = program_create(args, GL_TRUE);
	if(!program) {
		fprintf(stderr, "%s\n", program_get_log());
	}
	program_args_release(args);
*/

#endif //PROGRAM_H


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

#include "Platform.h"

// Utility functions
#if defined(_WIN32)

void ErrorMsg(const char *string){
	MessageBoxA(NULL, string, "Error", MB_OK | MB_ICONERROR);
}

void WarningMsg(const char *string){
	MessageBoxA(NULL, string, "Warning", MB_OK | MB_ICONWARNING);
}

void InfoMsg(const char *string){
	MessageBoxA(NULL, string, "Information", MB_OK | MB_ICONINFORMATION);
}

double inv_freq;

void initTime(){
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	inv_freq = 1.0 / double(freq.QuadPart);
}

timestamp getCurrentTime(){
	LARGE_INTEGER curr;
	QueryPerformanceCounter(&curr);
	return curr;
}

float getTimeDifference(const timestamp from, const timestamp to)
{
	int64 diff = to.QuadPart - from.QuadPart;
	return (float) (double(diff) * inv_freq);
}

#else

#if defined(LINUX)

#include <gtk/gtk.h>

// This is such a hack, but at least it works.
gboolean idle(gpointer data){
	gtk_main_quit();
	return FALSE;
}

void MessageBox(const char *string, const GtkMessageType msgType){
	GtkWidget *dialog = gtk_message_dialog_new(NULL, GTK_DIALOG_DESTROY_WITH_PARENT, msgType, GTK_BUTTONS_OK, string);
	gtk_dialog_run(GTK_DIALOG(dialog));
	gtk_widget_destroy(dialog);
	g_idle_add(idle, NULL);
	gtk_main();
}

void ErrorMsg(const char *string){
	MessageBox(string, GTK_MESSAGE_ERROR);
}

void WarningMsg(const char *string){
	MessageBox(string, GTK_MESSAGE_WARNING);
}

void InfoMsg(const char *string){
	MessageBox(string, GTK_MESSAGE_INFO);
}

#elif defined(__APPLE__)

void ErrorMsg(const char *string){
	Str255 msg;
	c2pstrcpy(msg, string);

	SInt16 ret;
	StandardAlert(kAlertStopAlert, msg, NULL, NULL, &ret);
}

void WarningMsg(const char *string){
	Str255 msg;
	c2pstrcpy(msg, string);

	SInt16 ret;
	StandardAlert(kAlertCautionAlert, msg, NULL, NULL, &ret);
}

void InfoMsg(const char *string){
	Str255 msg;
	c2pstrcpy(msg, string);

	SInt16 ret;
	StandardAlert(kAlertNoteAlert, msg, NULL, NULL, &ret);
}

#endif

void initTime(){
}

timestamp getCurrentTime(){
	timeval curr;
	gettimeofday(&curr, NULL);
	return curr;
}

float getTimeDifference(const timestamp from, const timestamp to)
{
	return (float(to.tv_sec - from.tv_sec) + 0.000001f * float(to.tv_usec - from.tv_usec));
}




#endif



#ifdef DEBUG

#include <stdio.h>

#ifdef _WIN32

void failedAssert(char *file, int line, char *statement){
	static bool debug = true;

	if (debug){
		char str[1024];

		sprintf(str, "Failed: (%s)\n\nFile: %s\nLine: %d\n\n", statement, file, line);

		if (IsDebuggerPresent()){
			strcat(str, "Debug?");
			int res = MessageBoxA(NULL, str, "Assert failed", MB_YESNOCANCEL | MB_ICONERROR);
			if (res == IDYES){
#if _MSC_VER >= 1400
				__debugbreak();
#else
				_asm int 0x03;
#endif
			} else if (res == IDCANCEL){
				debug = false;
			}
		} else {
			strcat(str, "Display more asserts?");
			if (MessageBoxA(NULL, str, "Assert failed", MB_YESNO | MB_ICONERROR | MB_DEFBUTTON2) != IDYES){
				debug = false;
			}
		}
	}
}

void outputDebugString(const char *str){
	OutputDebugStringA(str);
	OutputDebugStringA("\n");
}

#else

void outputDebugString(const char *str){
	printf("%s\n", str);
}

#endif // _WIN32
#endif // DEBUG

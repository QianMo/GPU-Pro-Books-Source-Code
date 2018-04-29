using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;


// Include Dir : "C:\Program Files\Microsoft DirectX SDK (August 2008)\Include";"$(SolutionDir)../Common"

// Libs : dxguid.lib d3dx9d.lib d3d9.lib d3dx10d.lib dinput8.lib winmm.lib comctl32.lib

// Lib dir : "C:\Program Files\Microsoft DirectX SDK (August 2008)\Lib\x86"

namespace Tool
{
	static class Program
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main()
		{
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Tool());

		}
	}
}

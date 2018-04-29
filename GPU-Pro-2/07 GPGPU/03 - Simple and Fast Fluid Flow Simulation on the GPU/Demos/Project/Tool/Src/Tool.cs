using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

using System.Threading;
using System.Reflection;

namespace Tool
{
    public partial class Tool : Form
	{
        
        private Timer           m_Timer;
        private Thread          m_Thread;

        private object SearchForInterface(String _DLLName, String _interfaceName)
        {
            Type ObjType = null;
            try
            {
                // Load it.
                Assembly ass = null;
                ass = Assembly.LoadFrom(_DLLName);
                if(ass!=null)
                {
                    Type[] types = ass.GetTypes();
                    foreach (Type t in types)
                    {
                        if (t != null)
                        {
                            if (t.GetInterface(_interfaceName) != null)
                            {
                                ObjType = t;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            try
            {
                if (ObjType != null)
                {
                    return Activator.CreateInstance(ObjType);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            return null;
            
        }

        /// <summary>
        /// ///
        /// </summary>
		public Tool()
		{
			InitializeComponent();

			IntPtr handle       = this.Handle;
            IntPtr PanelHandle  = splitContainer1.Panel1.Handle;
            Int32 _Width        = splitContainer1.Panel1.Width;
            Int32 _Height       = splitContainer1.Panel1.Height;

            String ViewerDLLName    = @"Viewer.dll";
            IRealTime iRT           = (IRealTime)SearchForInterface(ViewerDLLName, "IRealTime");

            if (iRT!=null)
            {
                iRT.Create(PanelHandle, _Width, _Height);
                this.MenuEnumVariables.DropDownItems.AddRange(iRT.GetMenuItems());
                this.LoadEnumVariables.DropDownItems.AddRange(iRT.GetLoadItems());
                splitContainer1.Panel1.MouseUp += iRT.GetMouseClick(); 
                
                
                m_Timer     = new Timer(iRT);
                m_Timer.SetNumUD(this.FrameRate);
                m_Thread    = new Thread(new ThreadStart(m_Timer.Run));
                m_Thread.Start();
                Closing += FermerFenetre;       
            }
		}

        /// <summary>
        /// <!-- Abort Render Thread-->
        /// <!-- Save changes       -->
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        public void FermerFenetre(object sender, System.ComponentModel.CancelEventArgs e)
        {
            
            if (MessageBox.Show("Do you want to save changes?", "Tool", MessageBoxButtons.YesNo) == DialogResult.Yes)
            {
                e.Cancel = true;
            }
            else
            {
                m_Thread.Abort();
                m_Thread.Join();
            }                   
        }

        private void MainMenu_Click(object _sender, EventArgs _e)
        {

        }

        private void LoadMenu_Click(object _sender, EventArgs _e)
        {

        }

		private void numericUpDown_ValueChanged(object sender, EventArgs e)
		{

		}




	}


}

namespace Tool
{
    partial class Tool
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{

            float QSize = 512;
            int Start   = 27;
            float CSize = 600;

            this.MenuVariables = new System.Windows.Forms.MenuStrip();
            this.MenuEnumVariables = new System.Windows.Forms.ToolStripMenuItem();
            this.LoadEnumVariables = new System.Windows.Forms.ToolStripMenuItem();
            this.FrameRate = new System.Windows.Forms.NumericUpDown();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.MenuVariables.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.FrameRate)).BeginInit();
            this.splitContainer1.SuspendLayout();
            this.SuspendLayout();
            // 
            // MenuVariables
            // 
            this.MenuVariables.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.MenuEnumVariables,
            this.LoadEnumVariables});
            this.MenuVariables.Location = new System.Drawing.Point(0, 0);
            this.MenuVariables.Name = "MenuVariables";
            this.MenuVariables.Size = new System.Drawing.Size((int)CSize, 24);
            this.MenuVariables.TabIndex = 0;
            this.MenuVariables.Text = "Variables Menu";
            // 
            // MenuEnumVariables
            // 
            this.MenuEnumVariables.BackColor = System.Drawing.SystemColors.ControlLight;
            this.MenuEnumVariables.Name = "MenuEnumVariables";
            this.MenuEnumVariables.Size = new System.Drawing.Size(50, 20);
            this.MenuEnumVariables.Text = "Menu";
            this.MenuEnumVariables.Click += new System.EventHandler(this.MainMenu_Click);
            // 
            // LoadEnumVariables
            // 
            this.LoadEnumVariables.BackColor = System.Drawing.SystemColors.ControlLight;
            this.LoadEnumVariables.Name = "LoadEnumVariables";
            this.LoadEnumVariables.Size = new System.Drawing.Size(45, 20);
            this.LoadEnumVariables.Text = "Load";
            this.LoadEnumVariables.Click += new System.EventHandler(this.LoadMenu_Click);
           
            // 
            // splitContainer1
            // 
            this.splitContainer1.Location = new System.Drawing.Point(0, Start);
            this.splitContainer1.Name = "splitContainer1";
            /// Fire Size,
            /// Size(1024, 768);


            this.splitContainer1.Size = new System.Drawing.Size((int)QSize + Start + 2, (int)QSize);
            this.splitContainer1.SplitterDistance = (int)QSize;
            this.splitContainer1.TabIndex = 2;

            // 
            // FrameRate
            ///<             
            this.FrameRate.DecimalPlaces = 12;
            this.FrameRate.Location = new System.Drawing.Point(12, (int)(QSize + 0.1f * QSize));
            this.FrameRate.Maximum = new decimal(new int[] {
            100000000,
            0,
            0,
            0});
            this.FrameRate.Name = "FrameRate";
            this.FrameRate.ReadOnly = true;
            this.FrameRate.Size = new System.Drawing.Size(120, 20);
            this.FrameRate.TabIndex = 4;
            this.FrameRate.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});

            // 
            // Tool
            // 
           // this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 6F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.None;// Font;
            this.ClientSize = new System.Drawing.Size((int)QSize + 5, (int)CSize);
            this.Controls.Add(this.splitContainer1);
            this.Controls.Add(this.FrameRate);
            this.Controls.Add(this.MenuVariables);
            this.MainMenuStrip = this.MenuVariables;
            this.Name = "Tool";
            this.Text = "Dev Tool";
            this.MenuVariables.ResumeLayout(false);
            this.MenuVariables.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.FrameRate)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

		}

		#endregion

        

		private System.Windows.Forms.MenuStrip          MenuVariables;
		private System.Windows.Forms.ToolStripMenuItem  MenuEnumVariables;


       // private System.Windows.Forms.MenuStrip          LoadVariables;
        private System.Windows.Forms.ToolStripMenuItem  LoadEnumVariables;


        private System.Windows.Forms.NumericUpDown      FrameRate;
        private System.Windows.Forms.SplitContainer     splitContainer1;
	}
}


using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Security;
using System.Security.Cryptography.X509Certificates;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace ROC_Tool
{
    public partial class Form1 : Form
    {
        public DataTable resultsTable = new DataTable();
        public DataTable confusionTable = new DataTable();
        public Series series1 = new Series();
        public Series series2 = new Series();
        public Series series3 = new Series();
        public Series series4 = new Series();
        public decimal threshold = 0.50M; //threshold as <double> created some precision problems 

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            confusionTable.Columns.Add("Cutoff", typeof(decimal));
            confusionTable.Columns.Add("TPR", typeof(double));
            confusionTable.Columns.Add("TNR", typeof(double));
            confusionTable.Columns.Add("FPR", typeof(double));
            confusionTable.Columns.Add("FNR", typeof(double));

            series1 = chart1.Series.Add("TPR");
            series1.ChartType = SeriesChartType.Line;
            series1.Color = Color.DarkBlue;
            series1.BorderWidth = 2;

            series2 = chart1.Series.Add("TNR");
            series2.ChartType = SeriesChartType.Line;
            series2.Color = Color.DarkOrange;
            series2.BorderWidth = 2;

            series3 = chart1.Series.Add("Threshold");
            series3.ChartType = SeriesChartType.Line;
            series3.Color = Color.Red;
            series3.BorderWidth = 1;

            series4 = chart1.Series.Add("Dots");
            series4.ChartType = SeriesChartType.Point;
            series4.Color = Color.Black;
            series4.BorderWidth = 3;

            chart1.ChartAreas[0].AxisX.Title = "Threshold Value";
            chart1.ChartAreas[0].AxisX.Minimum = 0;
            chart1.ChartAreas[0].AxisX.Maximum = 1;

            chart1.ChartAreas[0].AxisY.Title = "True Pos/Neg Rate";
            chart1.ChartAreas[0].AxisY.Minimum = -0.05;
            chart1.ChartAreas[0].AxisY.Maximum = 1.05;

            chart1.Legends[0].Enabled = false;
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                resultsTable.Columns.Add("Proba", typeof(double));
                resultsTable.Columns.Add("Actual", typeof(int));
                try
                {
                    using (StreamReader reader = new StreamReader(openFileDialog1.FileName))
                    {
                        bool isFirstRow = true;
                        while (!reader.EndOfStream)
                        {
                            var line = reader.ReadLine();
                            var values = line.Split(',');

                            if (isFirstRow)
                            {
                                //Think about creating headers from the file
                                isFirstRow = false;
                            }
                            else
                            {
                                resultsTable.Rows.Add(values);
                            }
                        }
                    }
                }
                catch (SecurityException ex)
                {
                    MessageBox.Show($"Security error.\n\nError message: {ex.Message}\n\n" +
                    $"Details:\n\n{ex.StackTrace}");
                }

                textBox1.Text = "Count: " + resultsTable.Rows.Count + "\r\n";
                int totalPos = resultsTable.AsEnumerable().Where(row => row.Field<int>("Actual") == 1).Count();
                textBox1.Text += "Positive: " + totalPos.ToString() + "\r\n";
                int totalNeg = resultsTable.AsEnumerable().Where(row => row.Field<int>("Actual") == 0).Count();
                textBox1.Text += "Negative: " + totalNeg.ToString() + "\r\n";

                for (double cutoff = 0.01; cutoff <= 0.99; cutoff += 0.01)
                {
                    var assertPos = resultsTable.AsEnumerable().Where(row => row.Field<double>("Proba") >= cutoff);  //We need this subtable NOT just its count 
                    int truePos = assertPos.AsEnumerable().Where(row => row.Field<int>("Actual") == 1).Count();
                    int falsePos = assertPos.Count() - truePos;

                    var assertNeg = resultsTable.AsEnumerable().Where(row => row.Field<double>("Proba") < cutoff);  //We need this subtable NOT just its count 
                    int trueNeg = assertNeg.AsEnumerable().Where(row => row.Field<int>("Actual") == 0).Count();
                    int falseNeg = assertNeg.Count() - trueNeg;

                    double TPR = (totalPos > 0) ? (double)truePos / totalPos : 1;
                    double TNR = (totalNeg > 0) ? (double)trueNeg / totalNeg : 1;
                    double FPR = (totalNeg > 0) ? (double)falsePos / totalNeg : 1;
                    double FNR = (totalPos > 0) ? (double)falseNeg / totalPos : 1;

                    textBox1.Text += "TPR: " + TPR.ToString("F5") + " TNR: " + TNR.ToString("F5") + "\r\n";

                    DataRow newRow = confusionTable.NewRow();
                    newRow.ItemArray = new object[] { cutoff, TPR, TNR, FPR, FNR };
                    confusionTable.Rows.Add(newRow);
                }
                foreach (DataRow row in confusionTable.Rows)
                {
                    series1.Points.AddXY(row["Cutoff"], row["TPR"]);
                    series2.Points.AddXY(row["Cutoff"], row["TNR"]);
                }
                trackBar1_Scroll(sender, e);
            }
        }
        private void fileToolStripMenuItem_Click(object sender, EventArgs e)
        {
        }
        private void openFileDialog1_FileOk(object sender, CancelEventArgs e)
        {
        }
        private void chart1_Click(object sender, EventArgs e)
        {
        }
        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            threshold = (decimal) trackBar1.Value / 100;
            textBox1.Text = threshold.ToString();
            chart1.Series["Threshold"].Points.Clear();
            for (double i = -0.05; i <= 1.05; i += 0.01)
            {
                series3.Points.AddXY(threshold, i);
            }
            var y1 = confusionTable.AsEnumerable().Where(row => row.Field<decimal>("Cutoff") == threshold);
            chart1.Series["Dots"].Points.Clear(); 
            foreach (DataRow row in y1)
            {
                series4.Points.AddXY(threshold, row["TPR"]);
                series4.Points.AddXY(threshold, row["TNR"]);
            }
        }
    }
}


using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using ScottPlot;
using ScottPlot.AxisPanels;

public class Program
{
    static void Main(string[] args)
    {
        string filename1 = "D:\\CPFxDRL-ZM12\\Generated_Airfoils\\XFOIL\\HELP12III1M.txt";
        string filename2 = "D:\\CPFxDRL-ZM12\\XFoil 6.99\\oa212.DAt";
        string outputImage = "D:\\CPFxDRL-ZM12\\Images\\III1M_vs_OA212..png";

        // first diagram data
        var xData1 = new List<double>();
        var yData1 = new List<double>();

        var xData2 = new List<double>();
        var yData2 = new List<double>();

        // --- ۳. تابع کمکی برای خواندن داده‌ها (برای جلوگیری از تکرار کد) ---
        // این تابع داده‌ها را می‌خواند و در لیست‌های X و Y ذخیره می‌کند
        void LoadDataFromFile(string path, List<double> xList, List<double> yList)
        {
            try
            {
                var lines = File.ReadAllLines(path);
                // ما منطق Skip(21) شما را برای هر دو فایل حفظ می‌کنیم
                foreach (var line in lines.Take(21).Concat(lines.Skip(21)))
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    var parts = line.Trim().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    if (parts.Length >= 2)
                    {
                        xList.Add(double.Parse(parts[0], CultureInfo.InvariantCulture));
                        yList.Add(double.Parse(parts[1], CultureInfo.InvariantCulture));
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading file {path}: {ex.Message}");
                throw; // خطا را به بیرون پرتاب می‌کنیم تا اجرای برنامه متوقف شود
            }
        }

        // --- ۴. خواندن داده‌ها برای هر دو فایل ---
        try
        {
            LoadDataFromFile(filename1, xData1, yData1);
            LoadDataFromFile(filename2, xData2, yData2);
        }
        catch
        {
            return;
        }


        Console.WriteLine($"Successfully loaded {xData1.Count} points from File 1 and 0 points from File 2.");

        // --- ۵. رسم نمودار با ScottPlot ---
        var plt = new ScottPlot.Plot();

        // **A. اضافه کردن و استایل‌دهی نمودار اول (YY.dat)**
        var scatter1 = plt.Add.Scatter(xData1, yData1);
        scatter1.LineColor = ScottPlot.Colors.Blue;
        scatter1.LineStyle.Pattern = ScottPlot.LinePattern.Solid;
        scatter1.MarkerShape = ScottPlot.MarkerShape.None;
        scatter1.LegendText = "HELP 12 III 1M";

        var scatter2 = plt.Add.Scatter(xData2, yData2);
        scatter2.LineColor = ScottPlot.Colors.Red;
        scatter2.LineStyle.Pattern = ScottPlot.LinePattern.Solid;
        scatter2.MarkerShape = ScottPlot.MarkerShape.None;
        scatter2.LegendText = "OA 212";

        // --- تنظیمات کلی نمودار ---
        //plt.Title("Resulted Airfoil");
        plt.XLabel("X/C");
        plt.YLabel("Y/C");
        //plt.Grid.IsVisible = true;
        //plt.Axes.SquareUnits();
        plt.Legend.BackgroundColor = Colors.Transparent;
        plt.Legend.OutlineColor = Colors.Transparent;

        // **C. اضافه کردن Legend (راهنما) برای نمایش لیبل‌ها**
        plt.ShowLegend();

        // تنظیم اندازه عکس هنگام ذخیره
        plt.ScaleFactor = 4;
        plt.Save(outputImage, 3200, 1000);

        Console.WriteLine($"Plot saved to {outputImage}");
    }
}

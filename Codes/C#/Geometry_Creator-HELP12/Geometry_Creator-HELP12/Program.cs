using System;
void GeomCreator()
{

    double x_Mu = 0.42761213;
    double y_Mu = 0.08;
    double x_Ml = 0.573656;
    double y_Ml = y_Mu - 0.12;

    double m1 = (1.0 - 0.3003177) / (1.0 - x_Mu);
    double n1 = (y_Mu) / (0.0600172);

    double m2 = (0.3003177) / (x_Mu);
    double n2 = (y_Mu) / (0.0600172);

    double m3 = (0.3003177) / (x_Ml);
    double n3 = (y_Ml) / (-.0600172);

    double m4 = (1.0 - 0.3003177) / (1.0 - x_Ml);
    double n4 = (y_Ml) / (-.0600172);

    double a0 = 0.052;
    double b4 = 0.09427006;
    double d0 = -0.052;

    double c4 = -1 * ((n2 * Math.Sqrt(m2)) / (n3 * Math.Sqrt(m3))) * b4;

    double a1 = -11.00000 * a0 + 0.59083;
    double a2 = 43.33333 * a0 - 2.08033;
    double a3 = -70.37037 * a0 + 3.06481;
    double a4 = 37.03701 * a0 - 1.57531;

    // suppress b0 because it is zero
    double b1 = -3.421459 * b4 + 0.531773;
    double b2 = 7.595197 * b4 - 1.545054;
    double b3 = -7.587163 * b4 + 1.464461;

    // suppress c0 because it is zero
    double c1 = -3.421459 * c4 - 0.531773;
    double c2 = 7.595197 * c4 + 1.545054;
    double c3 = -7.587163 * c4 - 1.464461;

    double d1 = -11.00000 * d0 - 0.59083;
    double d2 = 43.33333 * d0 + 2.08033;
    double d3 = -70.37037 * d0 - 3.06481;
    double d4 = 37.03701 * d0 + 1.57531;

    double[] x = new double[201];
    double[] y = new double[201];

    double f1(double _x)
    {

        double answer = 0;
        answer = a0 + (a1 * _x) + (a2 * Math.Pow(_x, 2)) + (a3 * Math.Pow(_x, 3)) + (a4 * Math.Pow(_x, 4));
        return answer;

    }
    double f2(double _x)
    {

        double answer = 0;
        answer = (b1 * _x) + (b2 * Math.Pow(_x, 2)) + (b3 * Math.Pow(_x, 3)) + (b4 * Math.Sqrt(_x));
        return answer;

    }
    double f3(double _x)
    {

        double answer = 0;
        answer = (c1 * _x) + (c2 * Math.Pow(_x, 2)) + (c3 * Math.Pow(_x, 3)) + (c4 * Math.Sqrt(_x));
        return answer;

    }
    double f4(double _x)
    {

        double answer = 0;
        answer = d0 + (d1 * _x) + (d2 * Math.Pow(_x, 2)) + (d3 * Math.Pow(_x, 3)) + (d4 * Math.Pow(_x, 4));
        return answer;

    }

    for (int i = 0; i < 101; i++)
    {
        double theta = i * Math.PI / 100;
        x[i] = 0.5 * (1 - Math.Cos(theta));
    }
    for (int i = 101; i < 201; i++)
    {
        double theta = (i - 100) * Math.PI / 100;
        x[i] = 0.5 * (1 - Math.Cos(theta));
    }

    //Upper Surface
    for (int i = 0; i < 101; i++)
    {

        //Segment 1
        if (x[i] >= x_Mu)
        {

            y[i] = y_Mu + n1 * (f1(0.3003177 + m1 * (x[i] - x_Mu)) - 0.0600172);

        }
        //Segment 2
        else if (x[i] < x_Mu)
        {

            y[i] = y_Mu + n2 * (f2(0.3003177 + m2 * (x[i] - x_Mu)) - 0.0600172);

        }

    }
    //Lower Surface
    for (int i = 101; i < 201; i++)
    {

        //Segment 3
        if (x[i] <= x_Ml)
        {

            y[i] = y_Ml + n3 * (f3(0.3003177 + m3 * (x[i] - x_Ml)) - (-.0600172));

        }
        //Segment 4
        else if (x[i] > x_Ml)
        {

            y[i] = y_Ml + n4 * (f4(0.3003177 + m4 * (x[i] - x_Ml)) - (-.0600172));

        }

    }

    y[100] = 0.0;
    y[200] = 0.0;

    string XFOIL_File_Path = "D:\\CPFxDRL-ZM12\\Generated_Airfoils\\XFOIL\\Created_Geometry.txt";
    string ANSYS_File_Path = "D:\\CPFxDRL-ZM12\\Generated_Airfoils\\ANSYS\\Created_Geometry.txt";
    WriteDatFile_XFOIL(XFOIL_File_Path, x, y);
    WriteDatFile_ANSYS(ANSYS_File_Path, x, y);

}

void WriteDatFile_XFOIL(string File_Path, double[] x, double[] y)
{

    using (var sw = new StreamWriter(File_Path))
    {
        //For using in ANSYS Space Claim
        //sw.WriteLine("3d=true");
        //sw.WriteLine("polyline=true");

        for (int i = 100; i >= 0; i--)
        {

            //Require format for ANSYS Space Claim (X   Y   Z)
            sw.WriteLine($"{x[i]} {y[i]}");

        }
        for (int i = 101; i < 201; i++)
        {

            //Require format for ANSYS Space Claim (X   Y   Z)
            sw.WriteLine($"{x[i]} {y[i]}");

        }

    }

}

void WriteDatFile_ANSYS(string File_Path, double[] x, double[] y)
{

    using (var sw = new StreamWriter(File_Path))
    {
        //For using in ANSYS Space Claim
        sw.WriteLine("3d=true");
        sw.WriteLine("polyline=true");

        for (int i = 100; i >= 0; i--)
        {

            //Require format for ANSYS Space Claim (X   Y   Z)
            sw.WriteLine($"{x[i]} {y[i]} 0.0");

        }
        for (int i = 101; i < 201; i++)
        {

            //Require format for ANSYS Space Claim (X   Y   Z)
            sw.WriteLine($"{x[i]} {y[i]} 0.0");

        }

    }

}

GeomCreator();

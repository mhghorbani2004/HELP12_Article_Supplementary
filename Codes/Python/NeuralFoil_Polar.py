from matplotlib import cm
import numpy as np
import neuralfoil as nf

def analyze_airfoil(input_file, output_file, Re=5e6):
    try:
        coords = np.loadtxt(input_file)
    except Exception as e:
        print("file not found")
        return

    alphas = np.linspace(0.0, 5.0, 51)

    aero_data = nf.get_aero_from_coordinates(
        coordinates=coords,
        alpha=alphas,
        Re=Re
        )

    CL = aero_data["CL"]
    CD = aero_data["CD"]
    CM = aero_data["CM"]

    data_to_save = np.column_stack((alphas, CL, CD, CM))

    header = "Alpha(deg)\tCL\tCD\tCM"

    np.savetxt(output_file, data_to_save, fmt="%.5f", delimiter="\t", header=header, comments="")

    print("File Saved")

if __name__ == "__main__":
    INPUT_TXT = r"D:\CPFxDRL-ZM12\Generated_Airfoils\XFOIL\HELP12III5M.txt"
    OUTPUT_TXT = r"D:\CPFxDRL-ZM12\NeuralFoil_Polars\HELP12III5M_NFPolar.txt"

    analyze_airfoil(INPUT_TXT, OUTPUT_TXT, Re=5e6)

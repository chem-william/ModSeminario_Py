import os.path
import numpy as np
from ase.units import Bohr
from ase.units import Hartree
from ase.units import kcal
from ase.units import mol
from ase.geometry.analysis import Analysis
from ase.io import read


def input_data_processing(inputfilefolder):
    """
    This function takes input data that is need from the files supplied
    Function extracts input coords and hessian from .fchk file, bond and angle
    lists from .log file and atom names if a z-matrix is supplied
    """

    # Gets Hessian in unprocessed form and writes .xyz file too
    unprocessed_Hessian, N, names, coords = coords_from_fchk(inputfilefolder, 'lig.fchk')

    # Gets bond and angle lists
    bond_list, angle_list = bond_angle_list(inputfilefolder)
    molecule = read(inputfilefolder + "input_coords.xyz")
    ana = Analysis(molecule)
    adjacency = ana.adjacency_matrix[0].toarray()
    adjacency = adjacency @ adjacency
    adjacency[np.diag_indices_from(adjacency)] = 0
    bond_list = np.argwhere(adjacency != 0)

    with open("Number_to_Atom_type") as f:
        OPLS_number_to_name = f.readlines()

    OPLS_number_to_name = [x.split() for x in OPLS_number_to_name]

    # Write the hessian in a 2D array format
    length_hessian = 3 * N
    hessian = np.zeros((length_hessian, length_hessian))
    m = 0
    for i in range(0, length_hessian):
        for j in range(0, i + 1):
            hessian[i][j] = unprocessed_Hessian[m]
            hessian[j][i] = unprocessed_Hessian[m]
            m = m + 1

    # Change Hartree/bohr -> kcal/mol/ang
    # hessian = (hessian * (Hartree * mol / kcal)) / Bohr**2
    hessian = hessian * Hartree / Bohr**2
    np.save(inputfilefolder + "hessian.npy", hessian)
    # if zmat exists part here
    atom_names = []

    for i in range(0, len(names)):
        atom_names.append(names[i].strip() + str(i + 1))

    if os.path.exists(inputfilefolder + 'Zmat.z'):
        atom_names = []

        fid = open(inputfilefolder + 'Zmat.z')  # Boss type Zmat

        tline = fid.readline()
        tline = fid.readline()

        # Find number of dummy atoms
        number_dummy = 0
        tmp = tline.split()

        while tmp[2] == '-1':
            number_dummy = number_dummy + 1
            tline = fid.readline()
            tmp = tline.split()

        if int(tmp[3]) < 800:
            for i in range(0, N):
                for j in range(0, len(OPLS_number_to_name)):
                    if OPLS_number_to_name[j][0] == tmp[3]:
                        atom_names.append(OPLS_number_to_name[j][1])

                tline = fid.readline()
                tmp = tline.split()
        else:
            # For CM1A format
            while len(tmp) < 2 or tmp[1] != 'Non-Bonded':
                tline = fid.readline()
                tmp = tline.split()

            tline = fid.readline()
            tline = fid.readline()
            tmp = tline.split()

            for i in range(0, N):
                atom_names.append(tmp[2])
                tline = fid.readline()
                tmp = tline.split()

        for i in range(0, N):
            if len(atom_names[i]) == 1:
                atom_names[i] = atom_names[i] + ' '

    return bond_list, angle_list, coords, N, hessian, atom_names


def coords_from_fchk(inputfilefolder, fchk_file):
    """
    Function extracts xyz file from the .fchk output file from Gaussian, this
    provides the coordinates of the molecules
    """
    if os.path.exists(inputfilefolder + fchk_file):
        fid = open((inputfilefolder + fchk_file), "r")
    else:
        with open((inputfilefolder + 'MSM_log'), "a") as fid_log:
            fid_log.write('ERROR = No .fchk file found.')
        return 0, 0

    tline = fid.readline()

    numbers = []  # Atomic numbers for use in xyz file
    list_coords = []  # List of xyz coordinates
    hessian = []

    # Get atomic number and coordinates from fchk
    while tline:
        # Atomic Numbers found
        if len(tline) > 16 and (tline[0:15].strip() == 'Atomic numbers'):
            tline = fid.readline()
            while (
                    len(tline) < 17 or
                    (tline[0:16].strip() != 'Nuclear charges')
            ):
                tmp = (tline.strip()).split()
                numbers.extend(tmp)
                tline = fid.readline()

        # Get coordinates
        if (
                len(tline) > 31 and
                tline[0:31].strip() == 'Current cartesian coordinates'
        ):
            tline = fid.readline()
            while (
                len(tline) < 15 or (
                    tline[0:14].strip() != 'Force Field' and
                    tline[0:17].strip() != 'Int Atom Types' and
                    tline[0:13].strip() != 'Atom Types' and
                    "Number of symbols in" not in tline.strip()
                )
            ):
                tmp = (tline.strip()).split()
                list_coords.extend(tmp)
                tline = fid.readline()
            N = int(float(len(list_coords)) / 3.0)  # Number of atoms

        # Gets Hessian
        if (
                len(tline) > 25 and
                (tline[0:26].strip() == 'Cartesian Force Constants')
        ):
            tline = fid.readline()

            while len(tline) < 13 or (tline[0:14].strip() != 'Dipole Moment'):
                tmp = (tline.strip()).split()
                hessian.extend(tmp)
                tline = fid.readline()

        tline = fid.readline()
    fid.close()

    list_coords = [float(x) * Bohr for x in list_coords]

    # Opens the new xyz file
    file = open(inputfilefolder + 'input_coords.xyz', "w")
    file.write(str(N) + '\n \n')

    xyz = np.zeros((N, 3))

    with open('elementlist.csv', "r") as fid_csv:
        lines = fid_csv.read().splitlines()

    # Turn list in a matrix,
    # with elements containing atomic number, symbol and name
    element_names = []
    for x in range(0, len(lines)):
        element_names.append(lines[x].split(","))

    # Gives name for atomic number
    names = []
    for x in range(0, len(numbers)):
        names.append(element_names[int(numbers[x]) - 1][1])

    # Print coordinates to new input_coords.xyz file
    n = 0
    for i in range(0, N):
        for j in range(0, 3):
            xyz[i][j] = list_coords[n]
            n = n + 1

        file.write(
            names[i] +
            str(round(xyz[i][0], 6)) +
            ' ' +
            str(round(xyz[i][1], 6)) +
            ' ' +
            str(round(xyz[i][2], 6)) +
            '\n'
        )
    file.close()

    return hessian, N, names, xyz


def bond_angle_list(inputfilefolder):
    """
    This function extracts a list of bond and angles from the Gaussian .log
    """
    fname = inputfilefolder + '/zmat.log'

    if os.path.isfile(fname):
        fid = open(fname, "r")
    elif os.path.isfile(inputfilefolder + '/lig.log'):
        fid = open((inputfilefolder + '/lig.log'), "r")
    else:
        with open((inputfilefolder + 'MSM_log'), "a") as fid_log:
            fid_log.write('ERROR - No .log file found. \n')
        return

    tline = fid.readline()
    bond_list = []
    angle_list = []

    tmp = 'R'  # States if bond or angle

    # Finds the bond and angles from the .log file
    while tline:
        tline = fid.readline()
        # Line starts at point when bond and angle list occurs
        if len(tline) > 80 and tline[0:81].strip() == '! Name  Definition              Value          Derivative Info.                !':
            tline = fid.readline()
            tline = fid.readline()
            # Stops when all bond and angles recorded
            while ((tmp[0] == 'R') or (tmp[0] == 'A')):
                line = tline.split()
                if len(line) < 2:
                    break
                tmp = line[1]

                # Bond or angles listed as string
                list_terms = line[2][2:-1]

                # Bond List
                if tmp[0] == 'R':
                    x = list_terms.split(',')
                    x = [(int(i) - 1) for i in x]
                    bond_list.append(x)

                # Angle List
                if tmp[0] == 'A':
                    x = list_terms.split(',')
                    x = [(int(i) - 1) for i in x]
                    angle_list.append(x)

                tline = fid.readline()

            # Leave loop
            tline = -1

    return bond_list, angle_list

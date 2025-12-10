import numpy as np
import math
import RF_Track as rft
import matplotlib.pyplot as plt

class Emitt_Meas_Simulation:
    def __init__(self, filename='Ext_ATF2/ATF2_EXT_FF_v5.2.twiss'):
        self.Pref=1.2999999e3
        self.filename=filename
        self.lattice = rft.Lattice(filename)
        self.sequence = [ e.get_name() for e in self.lattice['*']]
        self.screens = [e.get_name() for e in self.lattice['*OTR*']]
        for s in self.lattice['*OTR*']:
            screen = rft.Screen()
            screen.set_name(s.get_name())
            s.replace_with(screen)

    def get_data_from_twiss_file(self):
        with open(self.filename, "r") as file:
            lines = [line.strip() for line in file if line.strip()]
        star_symbol = next(i for i, line in enumerate(lines) if line.startswith("*"))
        dollar_sign = next(i for i, line in enumerate(lines) if line.startswith("$") and i > star_symbol)
        columns = lines[star_symbol].lstrip("*").split()
        BETA_X_column = columns.index("BETX") # m
        BETA_Y_column = columns.index("BETY") # m
        ALPHA_X_column = columns.index("ALFX")
        ALPHA_Y_column = columns.index("ALFY")
        MU_X_column = columns.index("MUX")
        MU_Y_column = columns.index("MUY")
        NAME_column = columns.index("NAME")
        names = []
        twiss = {}

        for line in lines[dollar_sign + 1:]:
            data = line.split()
            name = data[NAME_column].strip('"')
            names.append(name)

            twiss[name] = {
                "betx": float(data[BETA_X_column]),
                "bety": float(data[BETA_Y_column]),
                "alpx": float(data[ALPHA_X_column]),
                "alpy": float(data[ALPHA_Y_column]),
                "mux": float(data[MU_X_column]),
                "muy": float(data[MU_Y_column]),
            }

        entrance_index = names.index("ATF2$START")
        entrance_name = names[entrance_index]
        entrance = twiss[entrance_name]
        otr_names = ['OTR0X', 'OTR1X', 'OTR2X', 'OTR3X']
        otrs = {name: twiss[name] for name in otr_names}
        return entrance_name, entrance, otrs

    def compute_transport_matrix(self):
        entrance_name, entrance, otrs = self.get_data_from_twiss_file()

        beta_x_0 = entrance["betx"]
        beta_y_0 = entrance["bety"]

        Mx_matrix_rows = []
        My_matrix_rows = []

        for otr in otrs:
            delta_mux = (otrs[otr]["mux"] - entrance["mux"])*2*math.pi
            delta_muy = (otrs[otr]["muy"] - entrance["muy"])*2*math.pi

            R11_x = np.sqrt((otrs[otr]["betx"])/beta_x_0) * (math.cos(delta_mux) + entrance["alpx"] * math.sin(delta_mux))
            R12_x = np.sqrt(beta_x_0 * otrs[otr]["betx"]) * math.sin(delta_mux)

            R33_y = np.sqrt((otrs[otr]["bety"])/beta_y_0) * (math.cos(delta_muy) + entrance["alpy"] * math.sin(delta_muy))
            R34_y = np.sqrt(beta_y_0 * otrs[otr]["bety"]) * math.sin(delta_muy)

            Mx_matrix_rows.append([R11_x**2, 2*R11_x*R12_x, R12_x**2])
            My_matrix_rows.append([R33_y**2, 2*R33_y*R34_y, R34_y**2])

        Mx = np.array(Mx_matrix_rows)
        My = np.array(My_matrix_rows)
        return Mx, My

    def compute_beam_matrix(self):
        entrance_name, entrance, otrs = self.get_data_from_twiss_file()
        mass = rft.electronmass
        beta_gamma = self.Pref / mass
        emitt_x = beta_gamma * 2e-3    # mm mrad
        emitt_y = beta_gamma * 1.18e-5 # mm mrad
        beta_x_0 = entrance["betx"]
        beta_y_0 = entrance["bety"]
        alpha_x_0 = entrance["alpx"]
        alpha_y_0 = entrance["alpy"]
        gamma_y_0 = (1+alpha_y_0**2)/beta_y_0
        gamma_x_0 = (1+alpha_x_0**2)/beta_x_0

        sigma_1 = emitt_x * beta_x_0
        sigma_2 = emitt_x * (-alpha_x_0)
        sigma_5 = emitt_x * gamma_x_0
        sigma_8 = emitt_y * beta_y_0
        sigma_9 = emitt_y * (-alpha_y_0)
        sigma_10 = emitt_y * gamma_y_0

        Sigma_xy_beam = {
            "sigma_1" : sigma_1,
            "sigma_2" : sigma_2,
            "sigma_5" : sigma_5,
            "sigma_8" : sigma_8,
            "sigma_9" : sigma_9,
            "sigma_10" : sigma_10,
        }
        return Sigma_xy_beam

    def setup_beam0(self):
        entrance_name, entrance, otrs = self.get_data_from_twiss_file()
        population = 2e10
        mass = rft.electronmass
        charge = -1
        Pref = self.Pref # 1.3 GeV/c
        beta_gamma = Pref / mass
        Twiss = rft.Bunch6d_twiss()
        Twiss.beta_x = entrance["betx"] # 6.848560987        # m
        Twiss.beta_y = entrance["bety"]  # 2.935758992         # m
        Twiss.alpha_x = entrance["alpx"] # 1.108024744
        Twiss.alpha_y = entrance["alpy"] # -1.907222942
        Twiss.emitt_x = beta_gamma * 2e-3    # mm mrad
        Twiss.emitt_y = beta_gamma * 1.18e-5 # mm mrad
        nParticles = 10000 # number of macroparticles
        B0 = rft.Bunch6d(mass, population, charge, Pref, Twiss, nParticles)
        return B0

    def measure_sigmas(self,return_bunches=False):
        otr_names = ['OTR0X', 'OTR1X', 'OTR2X', 'OTR3X']
        otr_names_upper = [n.upper() for n in otr_names]

        B0 = self.setup_beam0()
        self.lattice.track(B0)
        B_screens = self.lattice.get_bunch_at_screens()

        sigma_x = []
        sigma_y = []

        for B in B_screens:
            info = B.get_info()
            sigma_x.append(info.sigma_x)
            sigma_y.append(info.sigma_y)

        sigma_x = np.array(sigma_x)
        sigma_y = np.array(sigma_y)
        screens_names=self.screens

        if return_bunches:
            return sigma_x,sigma_y, B_screens, screens_names
        else:
            return sigma_x, sigma_y

    def get_bunch_at_otr(self,otr_name):
        sigma_x, sigma_y, B_screens, inserted_screens = self.measure_sigmas(return_bunches=True)
        name_upper=otr_name.upper()
        idx = inserted_screens.index(name_upper)
        return B_screens[idx]


    def solve_least_squares(self,Mx, My, sigma_x_i, sigma_y_i):
        sigma_x2 = sigma_x_i **2
        sigma_y2 = sigma_y_i **2

        sigma_x_meas,*_ = np.linalg.lstsq(Mx, sigma_x2, rcond=None)
        sigma_y_meas, *_ = np.linalg.lstsq(My, sigma_y2, rcond=None)

        sigma1,sigma2,sigma5 = sigma_x_meas
        sigma8,sigma9,sigma10 = sigma_y_meas

        emittance_x = np.sqrt(np.abs(sigma1 * sigma5-sigma2**2))
        emittance_y = np.sqrt(np.abs(sigma8 * sigma10 - sigma9**2))

        beta_x = sigma1 / emittance_x
        beta_y = sigma8 / emittance_y

        alpha_x = -sigma2 / emittance_x
        alpha_y = -sigma9 / emittance_y

        gamma_x = sigma5 / emittance_x
        gamma_y = sigma10 / emittance_y

        return emittance_x, emittance_y, beta_x, beta_y, alpha_x, alpha_y, gamma_x, gamma_y

    def _plot_vertical_beam_size(self):
        B0 = self.setup_beam0()
        self.lattice.track(B0)
        L=self.lattice
        lattice_length = L.get_length()
        print("Lattice length:", lattice_length)
        T = L.get_transport_table('%S %sigma_y')
        plt.plot(T[:, 0], T[:, 1], 'o', linewidth=2, )
        plt.xlabel('s [m]')
        plt.ylabel(r'$\sigma_y$ [mm]')
        plt.show()

    def _plot_horizontal_beam_size(self):
        B0 = self.setup_beam0()
        self.lattice.track(B0)

        L=self.lattice
        lattice_length = L.get_length()
        print("Lattice length:", lattice_length)
        T = L.get_transport_table('%S %sigma_x')
        plt.plot(T[:, 0], T[:, 1], 'o', linewidth=2, )
        plt.xlabel('s [m]')
        plt.ylabel(r'$\sigma_x$ [mm]')
        plt.show()

    def _plot_normalised_horizontal_phase_space(self, otr_name):
        B = self.get_bunch_at_otr(otr_name)
        M = B.get_phase_space('%x %xp')
        plt.figure()
        plt.plot(M[:, 0], M[:, 1], '.', label=f'Phase space at {otr_name}', mfc='none')
        plt.xlabel('x [mm]')
        plt.ylabel("x' [mrad]")
        plt.legend()
        plt.show()
    def _plot_normalised_vertical_phase_space(self,otr_name):
        B = self.get_bunch_at_otr(otr_name)
        M = B.get_phase_space('%y %yp')
        plt.figure()
        plt.plot(M[:, 0], M[:, 1], '.', label=f'Phase space at {otr_name}', mfc='none')
        plt.xlabel('y [mm]')
        plt.ylabel("y' [mrad]")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    w = Emitt_Meas_Simulation()

    entrance_name, entrance, otrs = w.get_data_from_twiss_file()
    Mx, My = w.compute_transport_matrix()
    Sigma_xy_beam = w.compute_beam_matrix()

    # rf track reconstruction:
    sigma_x_i, sigma_y_i = w.measure_sigmas()
    emittance_x, emittance_y, beta_x, beta_y, alpha_x, alpha_y, gamma_x, gamma_y = w.solve_least_squares(Mx, My, sigma_x_i, sigma_y_i)

    w._plot_vertical_beam_size()
    w._plot_horizontal_beam_size()
    w._plot_normalised_horizontal_phase_space(otr_name = 'OTR1X')
    w._plot_normalised_vertical_phase_space(otr_name = 'OTR1X')
    mass = rft.electronmass
    Pref = w.Pref
    beta_gamma = Pref / mass
    print(f"Beta gamma is: {beta_gamma}")
    print(f"emittance_x (normalised) = {emittance_x} mm mrad")
    print(f"emittance_y (normalised) = {emittance_y} mm mrad")
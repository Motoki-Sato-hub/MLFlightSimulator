import numpy as np
import matplotlib.pyplot as plt
import RF_Track as rft
from scipy.optimize import minimize
import re

#not every monitor is a bpm
#BTV are screens, ICT are charge monitors
#only BPMS/BPC are actual bpms

def get_ITF(I):
    return 1.29404711e-2  - 2.59458259e-07*I # T/A

def get_grad(I, Lquad=0.226):
    G_0 = I * get_ITF(I) / Lquad    # T/m
    return G_0

def get_Quad_K(G_0, Pref):
    K = 299.8 *G_0 / Pref  # 1/m^2
    return K

def get_Quad_K_from_I(I, Lquad, Pref):
    G_0 = get_grad(I, Lquad)
    K = get_Quad_K(G_0, Pref)
    return K

class InterfaceCLEAR_RFTrack:
    def get_name(self):
        return 'CLEAR_RFT'

    # clear.survey0_filtered.tfs    CLEAR_Beamline_Survey.txt
    def obtaining_the_lattice(self, filename):
        with open(filename) as file:
            lines = file.readlines()
        element_descriptions = {}
        previous_name = None
        quad_index = 0
        corr_index = 0
        for line in lines:
            if line[0:2] != ' "':
                continue

            text = re.findall(r'"([A-Za-z0-9.$_]+)"', line)
            numbers = re.findall(r'\d+\.\d+', line)

            name = text[0]

            if name == 'CA.BTV0800':
                continue
            element_type = None
            if 'QFD' in name or 'QDD' in name:
                element_type = 'Quadrupole'
            elif 'BTV' in name:
                element_type = 'Screen'
            elif 'DHG' in name or 'DHJ' in name or 'SDV' in name:
                element_type = 'Corrector'
            elif 'BPC' in name or 'BPM' in name:
                element_type = 'BPM'
            elif len(text) > 1 and text[1] == 'MARKER':
                element_type = 'Marker'

            if element_type is None:
                continue

            s_end = float(numbers[0])
            L = float(numbers[1])
            s_start = s_end - L

            L = round(L, 4)
            s_start = round(s_start, 4)
            s_end = round(s_end, 4)

            if previous_name is not None:
                L_drift = round(s_start - element_descriptions[previous_name]['s_end'], 4)
                if L_drift != 0:
                    element_descriptions[previous_name + ' Drift'] = {
                        'element_type': 'Drift',
                        'L': L_drift,
                        's_start': element_descriptions[previous_name]['s_end'],
                        's_end': s_start,
                        'quad_index': None,
                        'corr_index': None,
                    }

            element_descriptions[name] = {
                'element_type': element_type,
                'L': L,
                's_start': s_start,
                's_end': s_end,
                'quad_index': quad_index if element_type == 'Quadrupole' else None,
                'corr_index': corr_index if element_type == 'Corrector' else None,
            }

            if element_type == 'Quadrupole':
                quad_index += 1
            if element_type == 'Corrector':
                corr_index += 1

            previous_name = name

        def get_lattice(start, end, Pref, quad_currents, include_end=True):
            start_index = list(element_descriptions.keys()).index(start)
            end_index = list(element_descriptions.keys()).index(end)
            if include_end:
                end_index += 1
            lattice = rft.Lattice()
            names = list(element_descriptions.keys())
            elements = list(element_descriptions.values())
            for name, element_description in zip(names[start_index:end_index], elements[start_index:end_index]):
                element_type = element_description['element_type']
                L = element_description['L']
                quad_index = element_description['quad_index']

                if element_type == 'Drift':
                    element = rft.Drift(L)
                elif element_type == 'Quadrupole':
                    if 'QFD' in name:
                        K = get_Quad_K_from_I(quad_currents[quad_index], L, Pref)
                    elif 'QDD' in name:
                        K = get_Quad_K_from_I(quad_currents[quad_index], L, Pref)
                    element = rft.Quadrupole(L, Pref / self.Q, K)
                elif element_type == 'Corrector':
                    element = rft.Corrector(L)
                elif element_type == 'BPM':
                    element = rft.Bpm(L)
                elif element_type == 'Screen' or element_type == 'Marker':
                    element = rft.Screen()
                else:
                    continue
                lattice.append(element)
            return lattice

        # Pref = self.Pref
        # quad_currents = np.array([
        #     0, 0, 0,
        #     0, 0, 0,
        #     0, 0, 0,
        #     0, 0
        # ])  # A

        # Here the lattice is constructed from the desired start to the desired end
        #L = get_lattice('CA.QFD0350', 'CA.BTV0910', Pref, quad_currents, include_end=False)
        start = 'CA.ACS0270S_MECH'
        end = 'CA.STLINE$END'
        lattice = get_lattice(start, end, self.Pref, np.array(11 * [0]))
        return lattice, element_descriptions, start, end

    def __init__(self, population=300 * rft.pC, jitter=0.0, bpm_resolution=0.0, nsamples=1):
        self.Pref = 198 # MeV/c
        self.Q=-1
        self.population = population
        self.jitter = jitter
        self.nsamples = nsamples
        self.mass=rft.electronmass
        self.lattice,self.element_descriptions,self.start,self.end=self.obtaining_the_lattice(filename="Interfaces/CLEAR/CLEAR_Beamline_Survey.txt")
        self.lattice.set_bpm_resolution(bpm_resolution)

        elements_in_lattice=list(self.lattice['*'])
        names_all=list(self.element_descriptions.keys())
        i0=names_all.index(self.start)
        i1=names_all.index(self.end)
        self.sequence=names_all[i0:i1]

        self._by_name=dict(zip(self.sequence,elements_in_lattice))
        self.bpms=[n for n in self.sequence if self.element_descriptions[n]['element_type'] == 'BPM']
        self.corrs=[n for n in self.sequence if self.element_descriptions[n]['element_type'] == 'Corrector']
        self.screens=[n for n in self.sequence if self.element_descriptions[n]['element_type'] == 'Screen']

        self.bpm_elements={n: self._by_name[n] for n in self.bpms}
        self.corrector_elements={n: self._by_name[n] for n in self.corrs}
        self.screen_elements={n: self._by_name[n] for n in self.screens}

        self.__setup_beam0()
        self.__track_bunch()
        self.freq=2.997e9
        self.nr_quad=11
        self.Lquad=0.226 #magnetic length of the quadrupole in [m]
        self.nominal_K=0.7752883624676146 #3.35  # 1/m
        self.Drift1 = rft.Drift(1.0)
        self.lattice.append(self.Drift1)
        self.lattice.get_length()
        # scr = next(iter(self.lattice._get_screens()))
        # print(type(scr))
        # print([m for m in dir(scr) if
        #        any(k in m.lower() for k in ["image", "read", "centroid", "profile", "mean", "sigma", "charge"])])
        # help(scr)

    def __setup_beam0(self): # (twiss params)they are the ones at the starting point of your constructed lattice
        T = rft.Bunch6d_twiss()
        T.emitt_x = 7.04 # mm.mrad normalised emittance
        T.emitt_y = 3.39  # 0.727 # mm.mrad
        T.beta_x = 15.6 # m
        T.beta_y = 24  #2.73  # m
        T.alpha_x = -0.49
        T.alpha_y = -3.65 #0.339
        T.sigma_t = 10*rft.ps #or 37*rft.ps # mm/c
        T.sigma_pt = 10 # permille
        T.mean_xp=0.0
        T.mean_yp=0.0
        Nparticles = 10000 # number of macroparticles
        self.P0 = rft.Bunch6d_QR(rft.electronmass, self.population, 1, self.Pref, T, Nparticles) # reference particle
        self.B0 = rft.Bunch6d_QR(rft.electronmass, self.population, - 1, self.Pref, T, Nparticles) # reference bunch
        # self.P0 = rft.Bunch6d(self.mass, self.population, self.Q, np.array([0,0,0,0,0,self.Pref]).T)
        # self.B0 = rft.Bunch6d(self.mass, self.population, self.Q, self.Pref, T, N)

    def __setup_beam1(self,scale):
        # Beam for DFS - Reduced energy
        Pref= scale * self.Pref
        T = rft.Bunch6d_twiss()
        T.emitt_x = 7.04 # mm.mrad normalised emittance
        T.emitt_y = 3.39  # 0.727 # mm.mrad
        T.beta_x = 15.6 # m
        T.beta_y = 24  #2.73  # m
        T.alpha_x = -0.49
        T.alpha_y = -3.65 #0.339
        T.sigma_t = 10*rft.ps #or 37*rft.ps # mm/c
        T.sigma_pt = 10 # permille
        T.mean_xp=0.0
        T.mean_yp=0.0
        Nparticles = 10000 # number of macroparticles
        self.B0 = rft.Bunch6d_QR(rft.electronmass, self.population, -1, Pref, T, Nparticles)
        self.P0 = rft.Bunch6d_QR(rft.electronmass, self.population,  1, Pref, T, Nparticles)

    def __setup_beam2(self,scale):
        # Beam for WFS - Reduced bunch charge
        population= scale * self.population
        T = rft.Bunch6d_twiss()
        T.emitt_x = 7.04 # mm.mrad normalised emittance
        T.emitt_y = 3.39  # 0.727 # mm.mrad
        T.beta_x = 15.6 # m
        T.beta_y = 24  #2.73  # m
        T.alpha_x = -0.49
        T.alpha_y = -3.65 #0.339
        T.sigma_t = 10*rft.ps #or 37*rft.ps # mm/c
        T.sigma_pt = 10 # permille
        T.mean_xp=0.0
        T.mean_yp=0.0
        Nparticles = 10000 # number of macroparticles
        self.B0 = rft.Bunch6d_QR(rft.electronmass, population, -1, self.Pref, T, Nparticles)
        self.P0 = rft.Bunch6d_QR(rft.electronmass, population,  1, self.Pref, T, Nparticles)

    def _get_screens(self):
        lattice,element_descriptions,start,end=self.obtaining_the_lattice(filename='CLEAR_Beamline_Survey.txt')
        highlight_filter = ['BTV390', 'BTV620', 'BTV730', 'BTV810', 'BTV910']  # <- short names
        highlight_positions = []
        highlight_names = []

        s = 0.0
        names = list(element_descriptions.keys())
        start_index = names.index(start)
        end_index = names.index(end) + 1
        names_in_lattice = names[start_index:end_index]

        for name in names_in_lattice:
            elem = element_descriptions[name]
            L = elem['L']
            elem_type = elem['element_type']

            if elem_type == 'Screen':
                short_name = name.split('.')[-1].replace('BTV0', 'BTV').replace('BTV', 'BTV')  # unify format
                if not highlight_filter or short_name in highlight_filter:
                    center = s + L / 2 if L > 0 else s
                    highlight_positions.append(center)
                    highlight_names.append(short_name)
            s += L

        print("Screens in lattice:")
        for name in names_in_lattice:
            elem = element_descriptions[name]
            if elem['element_type'] == 'Screen':
                print(name, "->", name.split('.')[-1], "at", elem['s_start'])

    def __track_bunch(self):
        I0 = self.B0.get_info()
        dx = self.jitter*I0.sigma_x
        dy = self.jitter*I0.sigma_y
        dz, roll = 0.0, 0.0
        pitch = self.jitter*I0.sigma_py
        yaw   = self.jitter*I0.sigma_px
        B0_offset = self.B0.displaced(dx, dy, dz, roll, pitch, yaw)
        self.lattice.track(B0_offset)

    def change_energy(self, scale):
        self.__setup_beam1(scale)
        self.__track_bunch()

    def reset_energy(self, scale=1):
        self.__setup_beam0( )
        self.__track_bunch()

    def change_intensity(self, scale): #reduced charge
        self.__setup_beam2(scale)
        self.__track_bunch()

    def reset_intensity(self, scale=1):
        self.__setup_beam0()
        self.__track_bunch()

    def get_sequence(self):
        return self.sequence

    def get_bpms_names(self):
        return self.bpms

    def get_correctors_names(self):
        return self.corrs

    def get_hcorrectors_names(self):
        return [string for string in self.corrs if "DHG" in string] #ITS PROBABLY WRONG

    def get_vcorrectors_names(self):
        return [string for string in self.corrs if ("DHJ" in string) or ("SDV" in string) ] #ITS PROBABLY WRONG

    def get_elements_position(self,names):
        return [index for index, string in enumerate(self.sequence) if string in names]

    def get_icts(self):
        print("Reading ict's...")
        charge = [ bpm.get_total_charge() for bpm in self.lattice.get_bpms() ]
        icts = {
            "names": self.bpms,
            "charge": charge
        }        
        return icts

    def get_correctors(self):
        print("Reading correctors' strengths...")
        bdes = np.zeros(len(self.corrs))
        for i,corrector in enumerate(self.corrs):
            c=self.corrector_elements[corrector]
            hx,hy=c.get_strength()
            if "DHG" in corrector: #horizontal
                bdes[i] = (hx*10)  # gauss*m
            elif ("SDV" in corrector) or ("DHJ" in corrector): #vertical
                bdes[i] = (hy*10)  # gauss*m
        correctors = { "names": self.corrs, "bdes": bdes, "bact": bdes }
        return correctors
    
    def get_bpms(self):
        print('Reading bpms...')
        nb=len(self.bpms)+len(self.screens)
        x = np.zeros((self.nsamples, nb))
        y = np.zeros(x.shape)
        tmit = np.zeros(x.shape)
        for i in range(self.nsamples):
            # for j,name in enumerate(self.bpms):
            #     b=self.bpm_elements[name]
            #     reading = b.get_reading()
            #     x[i,j] = reading[0]
            #     y[i,j] = reading[1]
            #     tmit[i,j] = b.get_total_charge()

            for j,name in enumerate(self.screens):
                s=self.screen_elements[name]
                B=s.get_bunch()
                I=B.get_info()
                x[i,j] = I.mean_x #is it ok? #mm
                y[i,j] = I.mean_y
                tmit[i,j] = B.get_total_charge()

        bpms = { "names": self.screens, "x": x, "y": y, "tmit": tmit }
        return bpms

    def push(self, names, corr_vals):
        if not isinstance(names, list):
            names = [ names ] # makes it a list
        for corr, val in zip(names, corr_vals):  #iteration of tuples
            c=self.corrector_elements[corr]
            if "DHG" in corr:
                c.set_strength(val/10, 0.0)  # T*mm
            elif ("DHJ" in corr) or ("SDV" in corr):
                c.set_strength(0.0, val/10)  # T*mm
        self.__track_bunch()
    
    def vary_correctors(self, names, corr_vals):
        if not isinstance(names, list):
            names = [ names ] # makes it a list
        for corr, val in zip(names, corr_vals):
            c=self.corrector_elements[corr]
            if "DHG" in corr:
                c.vary_strength(val/10, 0.0)  # T*mm
            elif ("SDV" in corr) or ("DHJ" in corr):
                c.vary_strength(0.0, val/10)  # T*mm
        self.__track_bunch()


#func for screens, return image etc


    #
    # def align_everything(self):
    #     self.lattice.align_elements()
    #     self.__track_bunch()
    #
    # def misalign_quadrupoles(self,sigma_x=0.100,sigma_y=0.100):
    #     self.lattice.scatter_elements('quadrupole', sigma_x, sigma_y, 0, 0, 0, 0, 'center')
    #     self.__track_bunch()
    #
    # def misalign_bpms(self,sigma_x=0.100,sigma_y=0.100):
    #     self.lattice.scatter_elements('bpm', sigma_x, sigma_y, 0, 0, 0, 0, 'center')
    #     self.__track_bunch()
    #

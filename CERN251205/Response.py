import numpy as np
import pickle

class Response():
    def __init__(self,filename=None):
        if filename is not None:
            self.load(filename)
        else:
            self.bpms = []
            self.hcorrs = []
            self.vcorrs = []
            self.Rxx = []
            self.Rxy = []
            self.Ryx = []
            self.Ryy = []
            self.Bx = []
            self.By = []

    def submatrix_B(self, bpms):
        bpm_indexes = [index for index, string in enumerate(self.bpms) if string in bpms]
        return (self.Bx[bpm_indexes],
                self.By[bpm_indexes])

    def submatrix_Rx(self, bpms, hcorrs):
        bpm_indexes = [index for index, string in enumerate(self.bpms) if string in bpms]
        hcorrs_indexes = [index for index, string in enumerate(self.hcorrs) if string in hcorrs]
        Rxx = self.Rxx[bpm_indexes,:]
        Ryx = self.Ryx[bpm_indexes,:]
        return (Rxx[:,hcorrs_indexes],
                Ryx[:,hcorrs_indexes])

    def submatrix_Ry(self, bpms, vcorrs):
        bpm_indexes = [index for index, string in enumerate(self.bpms) if string in bpms]
        vcorrs_indexes = [index for index, string in enumerate(self.vcorrs) if string in vcorrs]
        Rxy = self.Rxy[bpm_indexes,:]
        Ryy = self.Ryy[bpm_indexes,:]
        return (Rxy[:,vcorrs_indexes],
                Ryy[:,vcorrs_indexes])

    def load(self,filename):
        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        self.bpms = data['bpms']
        self.hcorrs = data['hcorrs']
        self.vcorrs = data['vcorrs']
        self.Rxx = data['Rxx']
        self.Rxy = data['Rxy']
        self.Ryx = data['Ryx']
        self.Ryy = data['Ryy']
        self.Bx = data['Bx']
        self.By = data['By']

    def save(self, filename):
        R = {
            "bpms": self.bpms,
            "hcorrs": self.hcorrs,
            "vcorrs": self.vcorrs,
            "Rxx": self.Rxx,
            "Rxy": self.Rxy,
            "Ryx": self.Ryx,
            "Ryy": self.Ryy,
            "Bx": self.Bx,
            "By": self.By
        }
        with open(filename, "wb") as pickle_file:
            pickle.dump(R, pickle_file)

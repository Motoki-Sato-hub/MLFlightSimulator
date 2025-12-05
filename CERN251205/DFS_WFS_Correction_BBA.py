import os, pickle, re, matplotlib, glob
import numpy as np
matplotlib.use("QtAgg")

class DFS_WFS_Correction_BBA():
    def _find_useful_files(self, directory):
        p_files = glob.glob(os.path.join(directory, "DATA_*_p*.pkl"))
        m_files = glob.glob(os.path.join(directory, "DATA_*_m*.pkl"))

        regex = re.compile(r"DATA_(.+)_(p|m)(\d+)\.pkl$")  # 3 groups - name corr, plus or minus, iteration
        p_index = {}
        m_index = {}

        for p in p_files:
            r = regex.search(os.path.basename(p))
            if r:
                corr, pm, iter = r.group(1), r.group(2), r.group(3)
                p_index[(corr, iter)] = p

        for m in m_files:
            r = regex.search(os.path.basename(m))
            if r:
                corr, pm, iter = r.group(1), r.group(2), r.group(3)
                m_index[(corr, iter)] = m

        valid_pairs = {}

        for (corr, iter), fp in p_index.items():
            fm = m_index.get((corr, iter))
            if fm:
                valid_pairs.setdefault(corr, []).append((fp, fm))  # dict for each corrector

        return {
            "ok": bool(valid_pairs),
            "dir": directory,
            "pairs": valid_pairs,
            "p_files": p_files,
            "m_files": m_files,
        }

    def _heaviside_function_for_checkbox(self, bpms, correctors):
        bpms_position = self.interface.get_elements_position(bpms)
        corrs_position = self.interface.get_elements_position(correctors)

        M = np.zeros((len(bpms), len(correctors)), dtype=bool)

        for j, cj in enumerate(corrs_position):
            for i, bi in enumerate(bpms_position):
                M[i, j] = (bi >= cj)

        return M

    def _get_data_from_loaded_directories(self, selected_bpms, selected_corrs, _force_triangular=False):

        info_traj = self._data_dirs["traj"]
        info_dfs = self._data_dirs["dfs"]
        info_wfs = self._data_dirs["wfs"]

        if not (info_traj and info_traj["ok"] and info_dfs and info_dfs["ok"] and info_wfs and info_wfs["ok"]):
            raise RuntimeError("Please select all data directories")

        hcorrs = [string for string in selected_corrs if
                  (string.lower().startswith('zh') or ("DHG" in string) or (string.lower().startswith('zx')))]
        vcorrs = [string for string in selected_corrs if
                  (string.lower().startswith('zv') or (("SDV" in string) or ("DHJ" in string)))]

        pairs0 = info_traj["pairs"]
        pairs1 = info_dfs["pairs"]
        pairs2 = info_wfs["pairs"]

        R0xx = np.full((len(selected_bpms), len(hcorrs)), np.nan, dtype=float)
        R0yy = np.full((len(selected_bpms), len(vcorrs)), np.nan, dtype=float)
        R0xy = np.zeros((len(selected_bpms), len(vcorrs)))
        R0yx = np.zeros((len(selected_bpms), len(hcorrs)))

        R1xx = np.full((len(selected_bpms), len(hcorrs)), np.nan, dtype=float)
        R1yy = np.full((len(selected_bpms), len(vcorrs)), np.nan, dtype=float)
        R1xy = np.zeros((len(selected_bpms), len(vcorrs)))
        R1yx = np.zeros((len(selected_bpms), len(hcorrs)))

        R2xx = np.full((len(selected_bpms), len(hcorrs)), np.nan, dtype=float)
        R2yy = np.full((len(selected_bpms), len(vcorrs)), np.nan, dtype=float)
        R2xy = np.zeros((len(selected_bpms), len(vcorrs)))
        R2yx = np.zeros((len(selected_bpms), len(hcorrs)))
        rows_B0x, rows_B0y = [], []

        pos = {b: i for i, b in enumerate(selected_bpms)}
        nb = len(selected_bpms)

        def _calculating_Rxx_or_Ryy(which_matrix, corrs_type, plane, pairs):

            for j, corr in enumerate(corrs_type):  # j is a column
                if corr not in pairs:
                    continue
                cols = []  # one column per plus/minus iteration
                for fp, fm in pairs[corr]:
                    with open(fp, "rb") as f:
                        plus_file = pickle.load(f)
                    with open(fm, "rb") as f:
                        minus_file = pickle.load(f)

                    bxp = np.asarray(plus_file["bpms"]["x"])  # turns into a vector (N,) instead of (N,1)
                    byp = np.asarray(plus_file["bpms"]["y"])
                    bxm = np.asarray(minus_file["bpms"]["x"])
                    bym = np.asarray(minus_file["bpms"]["y"])

                    bact_p = np.asarray(plus_file["correctors"]["bact"]).squeeze()  # bact is an actual corrector value/kick that was applied
                    bact_m = np.asarray(minus_file["correctors"]["bact"]).squeeze()

                    if bxp.ndim>1:
                        bxp=bxp.mean(axis=0)
                        bxm=bxm.mean(axis=0)
                        byp=byp.mean(axis=0)
                        bym=bym.mean(axis=0)

                    bpms_names_real = plus_file["bpms"]["names"]
                    if isinstance(bpms_names_real, list):
                        if len(bpms_names_real) > 0 and isinstance(bpms_names_real[0], list):
                            bpms_names = [str(b) for b in bpms_names_real[0]]
                        else:
                            bpms_names = [str(b) for b in bpms_names_real]
                    else:
                        bpms_names = [str(bpms_names_real)]
                    present=[b for b in selected_bpms if b in bpms_names]
                    if not present:
                        continue
                    indeces = [bpms_names.index(b) for b in present]
                    max_idx=max(indeces)
                    if plane=="x":
                        if max_idx >= len(bxp):
                            continue
                        plus_value = bxp[indeces]
                        minus_value = bxm[indeces]
                    elif plane == "y":
                        if max_idx >= len(byp):
                            continue
                        plus_value = byp[indeces]
                        minus_value = bym[indeces]

                    corrs_names_real = plus_file["correctors"]["names"]
                    if isinstance(corrs_names_real, list):
                        if len(corrs_names_real) > 0 and isinstance(corrs_names_real[0], list):
                            corrs_names = [str(c) for c in corrs_names_real[0]]
                        else:
                            corrs_names = [str(c) for c in corrs_names_real]
                    else:
                        corrs_names = [str(corrs_names_real)]

                    if corr not in corrs_names:
                        continue

                    i_corr = corrs_names.index(corr)
                    k_plus = float(bact_p[i_corr])
                    k_minus = float(bact_m[i_corr])


                    if pairs == pairs0 and plane == "x":
                        px = bxp[indeces]
                        mx = bxm[indeces]
                        B0_x = (px + mx) / 2  # golden orbit
                        rowx = np.full(nb, np.nan)
                        for k, b in enumerate(present):
                            rowx[pos[b]] = B0_x[k]
                        rows_B0x.append(rowx)

                        py = byp[indeces]
                        my = bym[indeces]
                        B0_y = (py + my) / 2
                        rowy = np.full(nb, np.nan)
                        for k, b in enumerate(present):
                            rowy[pos[b]] = B0_y[k]
                        rows_B0y.append(rowy)

                    if k_plus - k_minus == 0:
                        continue
                    column_value = np.full(len(selected_bpms), np.nan, dtype=float)
                    for k, b in enumerate(present):
                        column_value[pos[b]] = (plus_value[k] - minus_value[k]) / (
                                    k_plus - k_minus)  # (bpm plus - bpm minus) /(kick plus - kick minus0
                    cols.append(column_value)

                if cols:
                    which_matrix[:, j] = np.nanmean(np.vstack(cols), axis=0)
            return which_matrix

        R0xx = _calculating_Rxx_or_Ryy(which_matrix=R0xx, corrs_type=hcorrs, plane="x", pairs=pairs0)
        R0yy = _calculating_Rxx_or_Ryy(which_matrix=R0yy, corrs_type=vcorrs, plane="y", pairs=pairs0)

        B0x = np.nanmean(np.vstack(rows_B0x), axis=0).reshape(-1, 1)
        B0y = np.nanmean(np.vstack(rows_B0y), axis=0).reshape(-1, 1)

        R1xx = _calculating_Rxx_or_Ryy(which_matrix=R1xx, corrs_type=hcorrs, plane="x", pairs=pairs1)
        R1yy = _calculating_Rxx_or_Ryy(which_matrix=R1yy, corrs_type=vcorrs, plane="y", pairs=pairs1)

        R2xx = _calculating_Rxx_or_Ryy(which_matrix=R2xx, corrs_type=hcorrs, plane="x", pairs=pairs2)
        R2yy = _calculating_Rxx_or_Ryy(which_matrix=R2yy, corrs_type=vcorrs, plane="y", pairs=pairs2)

        if self._force_triangular() or _force_triangular:
            corrs, bpms = selected_corrs, selected_bpms
            Cx = [s for s in corrs if (s.lower().startswith('zh') or ("DHG" in s) or (s.lower().startswith('zx')))]
            Cy = [s for s in corrs if (s.lower().startswith('zv') or (("SDV" in s) or ("DHJ" in s)))]

            Mx = self._heaviside_function_for_checkbox(correctors=Cx, bpms=bpms)
            My = self._heaviside_function_for_checkbox(correctors=Cy, bpms=bpms)

            R0xx = np.where(Mx, R0xx, 0.0)  # builds new array, R0xx shape, if Mx is true otherwise 0.0
            R1xx = np.where(Mx, R1xx, 0.0)
            R2xx = np.where(Mx, R2xx, 0.0)

            R0yy = np.where(My, R0yy, 0.0)
            R1yy = np.where(My, R1yy, 0.0)
            R2yy = np.where(My, R2yy, 0.0)

        return R0xx, R0yy, R0xy, R0yx, R1xx, R1yy, R1xy, R1yx, R2xx, R2yy, R2xy, R2yx, B0x, B0y

    def _creating_response_matrices(self):

        w1, w2, w3, rcond, iters, gain = self._read_params()
        wgt_orb, wgt_dfs, wgt_wfs = w1, w2, w3

        corrs, bpms = self._get_selection()
        R0xx, R0yy, R0xy, R0yx, R1xx, R1yy, R1xy, R1yx, R2xx, R2yy, R2xy, R2yx, B0x, B0y = self._get_data_from_loaded_directories(
            selected_corrs=corrs, selected_bpms=bpms, _force_triangular=self._force_triangular())

        R0 = np.block([
            [R0xx, R0xy],
            [R0yx, R0yy],
        ])

        R1 = np.block([
            [R1xx, R1xy],
            [R1yx, R1yy],
        ])

        R2 = np.block([
            [R2xx, R2xy],
            [R2yx, R2yy],
        ])

        R_nom = R0
        R_disp = R1 - R0
        R_wake = R2 - R0

        Axx=[]
        Ayy=[]

        if wgt_orb > 0:
            Axx.append(wgt_orb * R0xx)
            Ayy.append(wgt_orb * R0yy)

        if wgt_dfs > 0:
            Axx.append(wgt_dfs * (R1xx - R0xx))
            Ayy.append(wgt_dfs * (R1yy - R0yy))

        if wgt_wfs > 0:
            Axx.append(wgt_wfs * (R2xx - R0xx))
            Ayy.append(wgt_wfs * (R2yy - R0yy))

        Axx = np.vstack(Axx)
        Ayy = np.vstack(Ayy)

        return Axx, Ayy, B0x, B0y

    def _get_dispersion_from_twiss_file(self):
        # madx
        corrs, bpms = self._get_selection()
        with open('Ext_ATF2/ATF2_EXT_FF_v5.2.twiss', "r") as file:
            # lines=file.readlines()
            lines = [line.strip() for line in file if line.strip()]

        star_symbol = next(i for i, line in enumerate(lines) if line.startswith("*"))
        dollar_sign = next(i for i, line in enumerate(lines) if line.startswith("$") and i > star_symbol)
        columns = lines[star_symbol].lstrip("*").split()

        DX_column = columns.index("DX")
        DY_column = columns.index("DY")
        elements_names = columns.index("NAME")

        bpms_dx, bpms_dy = {}, {}
        for line in lines[dollar_sign + 1:]:
            data = line.split()
            bpms_name = data[elements_names].strip('"')

            bpms_dx[bpms_name] = float(data[DX_column])
            bpms_dy[bpms_name] = float(data[DY_column])

        target_disp_x = np.array([bpms_dx.get(bpm, 0.0) for bpm in bpms]).reshape(-1, 1)  # m
        target_disp_y = np.array([bpms_dy.get(bpm, 0.0) for bpm in bpms]).reshape(-1, 1)  # m

        return target_disp_x, target_disp_y

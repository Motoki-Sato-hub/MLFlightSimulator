# FlightSimulator's `Interface` description

This class defines how the user interacts with the real or simulated machine when reading beam instrumentation and steering devices and for writing corrector set-points in a control system for the specific accelerator facility. It wraps common operations on BPMs, ICTs, and corrector magnets.

---

## Constructor

### `__init__(nsamples: int = 1)`

Initializes the interface and builds internal lists of devices (sequence, BPMs, correctors) and EPICS-like indexes.

**Parameters**

* `nsamples` — number of consecutive samples to take when reading BPMs (with a 1 s pause between samples).

---

## Identification

### `get_name() -> str`

Returns a short identifier of this interface: e.g., `"ATF2_Ext"`.

---

## Lattice & device inventory

### `get_sequence() -> list[str]`

Returns the full **beamline-ordered** list of device names (BPMs and correctors).

### `get_bpms_names() -> list[str]`

Returns the list of BPM names known to EPICS and present in the configured sequence.

### `get_correctors_names() -> list[str]`

Returns the list of corrector names (both planes) present in the configured sequence.

### `get_hcorrectors_names() -> list[str]`

Returns the subset of correctors that are **horizontal** (names starting with `ZH`).

### `get_vcorrectors_names() -> list[str]`

Returns the subset of correctors that are **vertical** (names starting with `ZV`).

### `get_elements_position(names: list[str] | str) -> list[int]`

Returns the **sequence indices** of the specified element name(s). Useful to map devices to their lattice order.

---

## Beam instrumentation reads

### `get_bpms() -> dict`

Reads BPM data from `LINAC:monitors` for `nsamples` shots, one per second.
**Returns** a dictionary:

* `"names"` — list of BPM names
* `"x"` — `(nsamples × NBPM)` array of **x** positions in **mm**
* `"y"` — `(nsamples × NBPM)` array of **y** positions in **mm**
* `"tmit"` — `(nsamples × NBPM)` array of intensity/charge (TMIT), **zeroed where BPM status ≠ 1**

> Note: Internally selects only those rows corresponding to `self.bpms` using `self.bpm_indexes`.

### `get_icts() -> dict`

Reads charge from all configured ICT PVs.
**Returns** a dictionary:

* `"names"` — list of ICT PV names
* `"charge"` — 1-D NumPy array of the corresponding readings

---

## Corrector reads & writes

### `get_correctors() -> dict`

Reads corrector strengths for all devices in `self.corrs`.
**Returns** a dictionary:

* `"names"` — list of corrector names
* `"bdes"` — desired/setpoint currents from `:<name>:currentWrite`
* `"bact"` — readback/actual currents from `:<name>:currentRead`

### `push(names: list[str] | str, corr_vals: list[float] | np.ndarray | float) -> None`

**Sets** the desired current for one or more correctors. If a single name/value is provided, it’s coerced to a list. Lengths must match. Writes to `:<name>:currentWrite` and sleeps 1 s.

### `vary_correctors(names: list[str] | str, corr_vals: list[float] | np.ndarray | float) -> None`

**Increments** the desired current for one or more correctors by the given value(s). Reads current value from `:<name>:currentWrite`, adds the increment, writes back, then sleeps 1 s.

---

## Placeholder

These are used by the BBA GUI. All of them re-track:

### `change_energy(scale: float) -> None`

Creates an off-energy bunch with Pref = scale * Pref_nom (e.g. 0.98) and tracks it.

### `reset_energy(scale: float) -> None`

Restores the nominal bunch (ignores scale and goes back to __setup_beam0()), then tracks.

### `change_intensity(scale: float) -> None`

Creates a low-charge bunch with population = scale * population_nom (e.g. 0.90) and tracks it.
### `reset_intensity(scale: float) -> None`

Restores the nominal bunch (ignores scale and goes back to __setup_beam0()), then tracks.

---

## Notes & best practices

* **Sampling**: Increase `nsamples` in the constructor if you want automatic multi-shot acquisition with per-shot 1 s spacing.
* **Units**: BPM positions are returned in **mm**; intensities are passed through the BPM status gate.
* **Safety**: `push` and `vary_correctors` talk directly to setpoint PVs. Use with care in live machines and follow your facility’s MPS/permit procedures.
* **Extensibility**: The placeholder energy/intensity methods can be filled with site-specific PVs or finite-state-machine hooks if needed.

## Notes for the DR

caget DR:monitors

DR:monitors

 array(i+0)  ... BPM(k) status (1=normal, others=error)
 array(i+1)  ... BPM(k) xpos
 array(i+2)  ... BPM(k) ypos
 array(i+3)  ... BPM(k) intensity
 array(i+4)  ... BPM(k) s position along the beam line 
 array(i+5)  ... BPM(k) name(0:3)
 array(i+6)  ... BPM(k) name(4:7)
 array(i+7)  ... BPM(k) name(8:11)
 array(i+8)  ... BPM(k) name(12:15)
 array(i+9)  ... BPM(k) reserved for future use
 array(i+10) ... BPM(k+1) status (1=normal, others=error)

i=k*10,
k= 0 ... MB1R    
 = 1 ... MB2R
 = 2 ... MB3R
 = 3 ... MB4R
 = 4 ... MB5R
 = 5 ... MB6R
 = 6 ... MB7R
 = 7 ... MB8R
 = 8 ... MB9R
 = 9 ... MB10R
 = 10 ... MB11R
 = 11 ... MB12R
 = 12,13,14 ... MB13R,MB14R,MB15R...
 = 97 ... MB98R(end)

 To change the mode of the DR

 caput DRBPM:ORBIT_MODE 1

 1 is for COD (averaging), the one needed
 2 is for one turn

 ## Notes for the mOTR

To insert/extract mOTR screens:
caput mOTR3:Target:WRITE:IN 1
caput mOTR3:Target:WRITE:OUT 1

To check the status of screen (in or out):
caget mOTR3:Target:READ:INOUT 

To Acquire images:

caget mOTR1:IMAGE:ArrayData
caput mOTR1:CAMERA:Acquire 1
caget mOTR1:IMAGE:ArrayData
Image size = 1280 x 960

To get the calibration factors

caget mOTR3:H:x1:Calibration:Factor


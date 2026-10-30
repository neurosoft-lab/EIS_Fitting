from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import json, math
import numpy as np

MODEL_ID = "geometry-resolved-mixed-conduction"
MATERIALS = ("DCH", "ap-DCH", "ICH")
STUDIES = ("Overlap", "Length", "Normalized interconnect area", "Width")


SPECS = {
    "rho_i": (1., 1e-6, 1e5, "ohm m"),
    "rho_e": (.1, 1e-8, 1e5, "ohm m"),
    "rc_au_pedot_area": (1e-7, 1e-12, 1e2, "ohm m^2"),
    "rct_overlap": (.01, 1e-9, 1e4, "ohm m^2"),
    "cdl_overlap": (.01, 1e-8, 1e3, "F/m^2"),
    "warburg_overlap": (.1, 1e-10, 1e6, "ohm m^2 s^-1/2"),
    "rct_bulk": (1e-6, 1e-14, 1e2, "ohm m^3"),
    "cdl_bulk": (1e4, 1e-4, 1e12, "F/m^3"),
    "warburg_bulk": (1e-4, 1e-14, 1e6, "ohm m^3 s^-1/2"),
    "rt_tip": (.01, 1e-9, 1e4, "ohm m^2"),
    "qt_tip": (.01, 1e-10, 1e7, "S s^n/m^2"),
    "nt_tip": (.7, .05, 1., "dimensionless"),
    "warburg_tip": (.1, 1e-10, 1e6, "ohm m^2 s^-1/2"),
    "bulk_pol_r_ref": (4e4, 1e1, 1e9, "ohm"),
    "bulk_pol_tau_ref": (5e-5, 1e-9, 1e3, "s"),
    "bulk_pol_geometry_exponent": (3., .2, 4., "dimensionless"),
    "bulk_pol_alpha": (.9, .2, 1., "dimensionless"),
}

def parameter_specs(material, options):

    keys = ["rho_i", "rct_overlap", "cdl_overlap", "warburg_overlap",
            "rt_tip", "qt_tip", "nt_tip", "warburg_tip"]
    if material != "ICH":
        keys += ["rho_e", "rc_au_pedot_area", "rct_bulk", "cdl_bulk", "warburg_bulk"]
    if options.get("bulk_concentration_polarization", False):
        keys += ["bulk_pol_r_ref", "bulk_pol_tau_ref", "bulk_pol_geometry_exponent", "bulk_pol_alpha"]
    return {key: SPECS[key] for key in keys}

def validate_geometry(g):
    if set(g) != {"overlap_length_um", "overlap_width_um", "thickness_um", "bulk"}:
        raise ValueError("Geometry requires overlap length, overlap width, thickness, and bulk sections.")
    for key in ("overlap_length_um", "overlap_width_um", "thickness_um"):
        if not np.isfinite(g[key]) or g[key] <= 0: raise ValueError(f"Invalid geometry value: {key}")
    if not g["bulk"]: raise ValueError("At least one bulk section is required.")
    for part in g["bulk"]:
        if set(part) != {"length_um", "width_start_um", "width_end_um", "region"}:
            raise ValueError("A bulk geometry section has missing or unknown fields.")
        for key in ("length_um", "width_start_um", "width_end_um"):
            if not np.isfinite(part[key]) or part[key] <= 0: raise ValueError(f"Invalid bulk value: {key}")
        if part["region"] not in ("body", "neck"): raise ValueError("Bulk region must be body or neck.")

@dataclass(frozen=True)
class Mesh:
    axial: np.ndarray
    overlap_area: np.ndarray
    bulk_volume: np.ndarray
    overlap_edges: int
    x_um: np.ndarray
    tip_area: float

def _part(length, w0, w1, height, step):
    n = max(1, math.ceil(length / step)); x = np.linspace(0., length, n + 1)
    if w0 != w1:
        count = max(1, math.ceil(abs(math.log(w1 / w0)) / .12))
        extra = length * (np.geomspace(w0, w1, count + 1) - w0) / (w1 - w0)
        x = np.sort(np.r_[x, extra]); x = x[np.r_[True, np.diff(x) > max(length, 1.) * 1e-12]]
        x[0], x[-1] = 0., length
    widths = w0 + (w1 - w0) * x / length; dx = np.diff(x) * 1e-6
    a0 = widths[:-1] * height * 1e-12; a1 = widths[1:] * height * 1e-12
    axial = dx / a0 if w0 == w1 else dx * np.log(a1 / a0) / (a1 - a0)
    return x, axial, dx * (a0 + a1) / 2

@lru_cache(maxsize=512)
def _mesh_cached(encoded, step):
    g = json.loads(encoded); validate_geometry(g)
    if not np.isfinite(step) or step <= 0: raise ValueError("Mesh step must be positive.")
    x, axial, _ = _part(g["overlap_length_um"], g["overlap_width_um"], g["overlap_width_um"], g["thickness_um"], step)
    overlap_edges = len(axial); contact = np.diff(x) * g["overlap_width_um"] * 1e-12
    intervals, areas, volumes, positions = [axial], [contact], [np.zeros(overlap_edges)], [x]
    offset = x[-1]
    for part in g["bulk"]:
        xp, gp, vp = _part(part["length_um"], part["width_start_um"], part["width_end_um"], g["thickness_um"], step)
        intervals.append(gp); areas.append(np.zeros(len(gp))); volumes.append(vp)
        positions.append(xp[1:] + offset); offset += xp[-1]
    axial = np.concatenate(intervals)
    if len(axial) > 12000: raise ValueError("A channel exceeds 12,000 elements; increase the mesh step.")
    def nodes(parts):
        values = np.concatenate(parts); out = np.zeros(len(values) + 1)
        out[:-1] += values / 2; out[1:] += values / 2
        return out
    mesh = Mesh(axial, nodes(areas), nodes(volumes), overlap_edges, np.concatenate(positions),
                g["bulk"][-1]["width_end_um"] * g["thickness_um"] * 1e-12)
    for value in vars(mesh).values():
        if isinstance(value, np.ndarray): value.setflags(write=False)
    return mesh

def prepare_mesh(g, step=100.): return _mesh_cached(json.dumps(g, sort_keys=True), float(step))

def concentration_polarization_impedance(p, mesh, frequency, options):

    f = np.asarray(frequency, float)
    if not options.get("bulk_concentration_polarization", False): return np.zeros(len(f), complex)
    integral = float(mesh.axial[mesh.overlap_edges:].sum()); reference = float(options["bulk_polarization_reference_axial_m_inv"])
    if integral <= 0 or reference <= 0: raise ValueError("Invalid concentration-polarization geometry.")
    scale = integral / reference; jw = 2j * np.pi * f
    return p["bulk_pol_r_ref"] * scale ** p["bulk_pol_geometry_exponent"] / (1 + (jw * p["bulk_pol_tau_ref"] * scale ** 2) ** p["bulk_pol_alpha"])

def ionic_edge_impedance(p, mesh, frequency, options):
    base = p["rho_i"] * mesh.axial; polarization = concentration_polarization_impedance(p, mesh, frequency, options)
    if not np.any(polarization): return base, polarization
    weights = mesh.axial[mesh.overlap_edges:].astype(float)
    result = np.broadcast_to(base, (len(polarization), len(base))).astype(complex).copy()
    result[:, mesh.overlap_edges:] += polarization[:, None] * (weights / weights.sum())[None, :]
    return result, polarization

def _edge(values, k): return values[k] if np.ndim(values) == 1 else values[:, k]

def grounded_three_rail_ladder(rung, re, ri, gold_contact):

    y = np.asarray(rung, complex); g = np.asarray(gold_contact, float)
    if y.ndim != 2 or y.shape[1] != len(re) + 1 or len(g) != y.shape[1]: raise ValueError("Three-rail dimensions are inconsistent.")
    a, b, c = g[0] + y[:, 0], y[:, 0].copy(), -y[:, 0]
    for k in range(len(re)):
        det = a * b - c * c
        if np.any(abs(det) < 1e-300): raise FloatingPointError("Singular three-rail admittance.")
        m00, m11, m01 = b / det + re[k], a / det + _edge(ri, k), -c / det
        dm = m00 * m11 - m01 * m01
        if np.any(abs(dm) < 1e-300): raise FloatingPointError("Singular three-rail impedance.")
        p00, p11, p01 = m11 / dm, m00 / dm, -m01 / dm; local = y[:, k + 1]
        a, b, c = p00 + g[k + 1] + local, p11 + local, p01 - local
    return 1 / (b - c * c / a)

def spectrum(material, p, g, frequency, options=None, rs=0., mesh=None):

    options = options or {}; f = np.asarray(frequency, float)
    if material not in MATERIALS: raise ValueError("Unknown material.")
    if f.ndim != 1 or not np.isfinite(f).all() or (f <= 0).any(): raise ValueError("Frequency must be a positive finite vector.")
    for key in parameter_specs(material, options):
        if key not in p or not np.isfinite(p[key]) or p[key] <= 0: raise ValueError(f"Missing or invalid parameter: {key}")
    if p["nt_tip"] > 1 or p.get("bulk_pol_alpha", 1) > 1: raise ValueError("CPE exponents cannot exceed 1.")
    if not np.isfinite(rs) or rs < 0: raise ValueError("Series resistance must be finite and non-negative.")
    mesh = mesh or prepare_mesh(g, options.get("mesh_step_um", 100.)); jw = 2j * np.pi * f; root = np.sqrt(jw)
    tip_series = p["rt_tip"] + p["warburg_tip"] / root
    ztip = 1 / (mesh.tip_area * (1 / tip_series + p["qt_tip"] * jw ** p["nt_tip"]))
    yo = 1 / (p["rct_overlap"] + p["warburg_overlap"] / root) + jw * p["cdl_overlap"]
    ri, zpol = ionic_edge_impedance(p, mesh, f, options)
    if material == "ICH":
        y = yo[:, None] * mesh.overlap_area[:mesh.overlap_edges + 1]; zov = 1 / y[:, 0]
        for k in range(mesh.overlap_edges): zov = 1 / (y[:, k + 1] + 1 / (_edge(ri, k) + zov))
        zbulk = np.full(len(f), p["rho_i"] * mesh.axial[mesh.overlap_edges:].sum(), complex)
        network = zov + zbulk + zpol
    else:
        yb = 1 / (p["rct_bulk"] + p["warburg_bulk"] / root) + jw * p["cdl_bulk"]
        rung = yo[:, None] * mesh.overlap_area + yb[:, None] * mesh.bulk_volume
        re = p["rho_e"] * mesh.axial; gold = mesh.overlap_area / p["rc_au_pedot_area"]
        network = grounded_three_rail_ladder(rung, re, ri, gold); zov = zbulk = None
    total = rs + network + ztip
    if not np.isfinite(total).all() or (total.real < -1e-7 * np.maximum(abs(total), 1)).any(): raise FloatingPointError("The network produced a non-finite or non-passive spectrum.")
    return total, {"Znetwork": network, "Ztip": ztip, "Zintrinsic": total, "Zoverlap": zov,
                   "ZbulkOhmic": zbulk, "ZbulkPolarization": zpol if options.get("bulk_concentration_polarization", False) else None}

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import time
import threading
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc
from scipy.sparse import csr_matrix
from .unified import MODEL_ID, MATERIALS, STUDIES, parameter_specs, prepare_mesh, spectrum, validate_geometry
from .run_control import RunControl,pause_checkpoint,active_time

DEFAULT_OPTIONS = dict(mesh_step_um=100., bulk_concentration_polarization=False,
                       bulk_polarization_reference_axial_m_inv=1e6)
DEFAULT_SEARCH = dict(starts=24, batches=2, short_nfev=100, max_nfev=2000, keep=5,
                      workers=4, seed=20260831, time_limit_seconds=0., resume_result='',
                      local_time_limit_seconds=0.)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timestamp(): return datetime.now().astimezone().isoformat()
def write_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def load_project(path):
    p = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    root = Path(path).resolve().parent
    for row in p.get('datasets', []):
        if not Path(row['data_dir']).is_absolute(): row['data_dir'] = str((root/row['data_dir']).resolve())
    for key in ('output_root','reference_result'):
        if p.get(key) and not Path(p[key]).is_absolute(): p[key] = str((root/p[key]).resolve())
    resume = p.get('search', {}).get('resume_result')
    if resume and not Path(resume).is_absolute(): p['search']['resume_result'] = str((root/resume).resolve())
    return validate_project(p)

def validate_project(project):
    p = deepcopy(project)
    if p.get('model_id') != MODEL_ID: raise ValueError('The project model does not match this application.')
    if not p.get('output_root'): raise ValueError('output_root is required.')
    unknown = set(p.get('options', {}))-set(DEFAULT_OPTIONS)
    if unknown: raise ValueError(f'Unknown model options: {unknown}')
    p['options'] = DEFAULT_OPTIONS | p.get('options', {})
    unknown = set(p.get('search', {}))-set(DEFAULT_SEARCH)
    if unknown: raise ValueError(f'Unknown search options: {unknown}')
    p['search'] = DEFAULT_SEARCH | p.get('search', {})
    if type(p['options']['bulk_concentration_polarization']) is not bool: raise ValueError('bulk_concentration_polarization must be Boolean.')
    for key in ('mesh_step_um','bulk_polarization_reference_axial_m_inv'):
        if not np.isfinite(p['options'][key]) or p['options'][key] <= 0: raise ValueError(f'Invalid option: {key}')
    for key in ('starts','batches','short_nfev','max_nfev','keep'):
        if not isinstance(p['search'][key], int) or p['search'][key] < 1: raise ValueError(f'{key} must be a positive integer.')
    if p['search']['workers'] < 0 or not np.isfinite(p['search']['time_limit_seconds']) or p['search']['time_limit_seconds'] < 0: raise ValueError('Invalid performance budget.')
    if not np.isfinite(p['search']['local_time_limit_seconds']) or p['search']['local_time_limit_seconds']<0:raise ValueError('Invalid local time budget.')
    if p.get('reference_mode', 'seed') not in ('seed','fixed_overlap'): raise ValueError('Invalid reference mode.')
    ids = set(); studies = {}
    if not p.get('datasets'): raise ValueError('The project has no datasets.')
    for row in p['datasets']:
        if not row.get('id') or row['id'] in ids or any(c in row['id'] for c in '/\\:'): raise ValueError('Dataset IDs must be unique and path-safe.')
        ids.add(row['id'])
        if row.get('material') not in MATERIALS or row.get('study') not in STUDIES: raise ValueError('Unsupported material or geometry study.')
        if type(row.get('training', True)) is not bool: raise ValueError('training must be Boolean.')
        if not isinstance(row.get('data_dir'), str): raise ValueError('data_dir is required.')
        row.setdefault('training', True); row.setdefault('batch', row['id']); row.setdefault('replicate', 1)
        if any(c in row['batch'] for c in '/\\:') or not row['batch']: raise ValueError('Invalid measurement batch name.')
        if row['replicate'] != 'all' and (not isinstance(row['replicate'], int) or row['replicate'] < 1): raise ValueError('replicate must be a positive integer or all.')
        if len(row.get('geometry', [])) != 8: raise ValueError(f"{row['id']} must contain eight channel geometries.")
        for g in row['geometry']: validate_geometry(g)

        encoded = json.dumps(row['geometry'], sort_keys=True)
        if row['study'] in studies and encoded != studies[row['study']]:

            if row['geometry'] != json.loads(studies[row['study']]): raise ValueError(f"Geometry differs between materials in {row['study']}.")
        studies[row['study']] = encoded
    return p

def read_spectrum(path):
    frame = pd.read_csv(path, sep=r'[\s,]+', engine='python', header=None, comment='#')
    values = frame.iloc[:, :3].apply(pd.to_numeric, errors='coerce')

    numeric_first = pd.to_numeric(frame.iloc[:, 0], errors='coerce').notna()
    positions=np.flatnonzero(numeric_first.to_numpy())
    if len(positions) and not numeric_first.iloc[positions[0]:].all():
        raise ValueError(f'Invalid frequency or text inside the numeric data block: {path}')
    if values[numeric_first].isna().any().any(): raise ValueError(f'Missing or invalid numeric data: {path}')
    a = values[numeric_first].to_numpy(float)
    if len(a)<3 or not np.isfinite(a).all() or (a[:,0]<=0).any(): raise ValueError(f'Invalid EIS file: {path}')
    a = a[np.argsort(a[:,0])]
    if len(np.unique(a[:,0])) != len(a): raise ValueError(f'Duplicate frequencies: {path}')
    z = a[:,1]-1j*a[:,2]
    if (abs(z)==0).any(): raise ValueError(f'Zero-impedance observation: {path}')
    return a[:,0], z

@dataclass
class Trace:
    dataset: str
    material: str
    batch: str
    channel: int
    replicate: int
    geometry: dict
    frequency: np.ndarray
    z: np.ndarray
    path: str
    digest: str
    balance: float = 1.

def load_traces(project, training=True):
    traces = []; manifest = []; duplicates = []; seen = {}
    import re
    for row in project['datasets']:
        if row['training'] != training: continue
        folder = Path(row['data_dir'])
        if not folder.is_dir(): raise ValueError(f'Data folder does not exist: {folder}')
        files = []
        for path in folder.glob('*.txt'):
            match = re.fullmatch(r'ch([1-8])-(\d+)\.txt', path.name, flags=re.I)
            if match and (row['replicate']=='all' or int(match[2])==row['replicate']):
                files.append((int(match[1]), int(match[2]), path))
        if {x[0] for x in files} != set(range(1,9)): raise ValueError(f"{row['id']} is missing one or more CH1-CH8 files.")
        local = []
        for ch, rep, path in sorted(files):
            digest = sha(path); f, z = read_spectrum(path); g = row['geometry'][ch-1]
            record = dict(dataset=row['id'], material=row['material'], channel=ch, replicate=rep,
                          path=str(path.resolve()), sha256=digest, frequency_points=len(f), training=training)
            manifest.append(record)


            key = (digest, row['material'], json.dumps(g, sort_keys=True))
            if key in seen:
                duplicates.append(dict(**record, duplicate_of=seen[key])); continue
            seen[key] = str(path)
            local.append(Trace(row['id'], row['material'], row['batch'], ch, rep, g, f, z, str(path), digest))
        for tr in local: tr.balance = 1/math.sqrt(max(len(local),1)*len(tr.frequency))
        traces += local
    return traces, manifest, duplicates

class Layout:
    def __init__(self, project, traces, anchors=None):
        self.project = project; self.entries = {}; self.anchors = anchors or {}
        self.materials = sorted({t.material for t in traces}); self.batches = sorted({t.batch for t in traces})
        for material in self.materials:
            for key, spec in parameter_specs(material, project['options']).items():
                self.entries[f'{material}/{key}'] = spec
        for batch in self.batches:
            self.entries[f'measurement/{batch}/Rs'] = (100., 1e-6, 1e8, 'ohm')
        self.fixed = {}; self.free = []; initial = []; lower = []; upper = []
        overrides = project.get('parameters', {})
        overrides = deepcopy(overrides)
        self.geometry_inactive = []
        if set(overrides)-set(self.entries): raise ValueError(f'Parameters do not apply to this project: {set(overrides)-set(self.entries)}')
        for name, (default, lo, hi, unit) in list(self.entries.items()):
            item = overrides.get(name, {})
            lo, hi = item.get('lower', lo), item.get('upper', hi)
            value = item.get('value', self.anchors.get(name, default))
            if not all(np.isfinite(x) for x in (value,lo,hi)) or not 0 < lo < hi or not lo <= value <= hi: raise ValueError(f'Invalid parameter or bounds: {name}')
            if name.endswith(('/nt_tip','/bulk_pol_alpha')) and hi>1: raise ValueError('CPE exponent upper bounds cannot exceed 1.')
            self.entries[name] = (value,lo,hi,unit)
            if item.get('fixed', False): self.fixed[name] = value
            else:
                self.free.append(name); initial.append(math.log(value)); lower.append(math.log(lo)); upper.append(math.log(hi))
        self.initial = np.asarray(initial); self.lower = np.asarray(lower); self.upper = np.asarray(upper)

    def unpack(self, x): return self.fixed | dict(zip(self.free, np.exp(x).tolist()))

    def material(self, values, material):
        p = {key.split('/',1)[1]: value for key, value in values.items() if key.startswith(material+'/')}
        return p

    def affects(self, name, trace):
        return name.startswith(trace.material+'/') or name.startswith(f'measurement/{trace.batch}/')

class Problem:
    def __init__(self, project, traces, layout):
        self.project = project; self.traces = traces; self.layout = layout
        self.meshes = [prepare_mesh(t.geometry, project['options']['mesh_step_um']) for t in traces]
        self._curve_cache = {}

    def curves(self, x, short=False):
        pause_checkpoint(getattr(self,'run_control',None))
        values = self.layout.unpack(x); curves = []
        for index, (tr, mesh) in enumerate(zip(self.traces, self.meshes)):
            ix = np.unique(np.linspace(0, len(tr.frequency)-1, min(24,len(tr.frequency))).round().astype(int)) if short else np.arange(len(tr.frequency))
            physical=self.layout.material(values,tr.material)
            rs=values[f'measurement/{tr.batch}/Rs'];key=(tuple(sorted(physical.items())),rs)
            cached=self._curve_cache.get((index,short))
            if cached is not None and cached[0]==key:z=cached[1]
            else:
                z, _ = spectrum(tr.material,physical,tr.geometry,tr.frequency[ix],self.project['options'],rs,mesh)
                self._curve_cache[index,short]=(key,z)
            curves.append((ix, z))
        return curves

    def residual(self, x, short=False):
        rows = []
        try:
            for tr, (ix, z) in zip(self.traces, self.curves(x, short)):
                balance = tr.balance*math.sqrt(len(tr.frequency)/len(ix))
                r = (z-tr.z[ix])/abs(tr.z[ix])*balance
                rows.extend((r.real,r.imag))
            return np.concatenate(rows)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            count = sum(2*min(24,len(t.frequency)) if short else 2*len(t.frequency) for t in self.traces)
            return np.full(count, 1e15)

    def sparsity(self, short=False):
        rows = []
        for tr in self.traces:
            n = min(24,len(tr.frequency)) if short else len(tr.frequency)
            rows.extend([list(self.layout.affects(name, tr) for name in self.layout.free)]*(2*n))
        return csr_matrix(rows, dtype=bool)

_PROBLEM = None
_CANCEL = None
_LIMITER = None
def _init_worker(problem, cancel=None):
    global _PROBLEM, _LIMITER, _CANCEL
    _PROBLEM = problem
    _CANCEL = cancel
    _PROBLEM.run_control=cancel if isinstance(cancel,RunControl) else None
    try:
        from threadpoolctl import threadpool_limits
        _LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _LIMITER = None

def _solve_local(task):
    x, short, budget, deadline = task
    problem = _PROBLEM
    stop_reason=None
    def callback(_):
        nonlocal stop_reason
        pause_checkpoint(_CANCEL)
        if _CANCEL is not None and _CANCEL.is_set():stop_reason='user_stopped';raise StopIteration
        if deadline and active_time(_CANCEL)>deadline:stop_reason='time_limit';raise StopIteration
    if not len(x):
        r = problem.residual(x); return dict(x=x, score=float(r@r), nfev=1)
    try:
        result = least_squares(lambda v: problem.residual(v, short),
            np.clip(x,problem.layout.lower+1e-10,problem.layout.upper-1e-10),
            bounds=(problem.layout.lower,problem.layout.upper),
            jac_sparsity=problem.sparsity(short) if len(x)>1 else None,
            x_scale='jac', max_nfev=budget, callback=callback)
        r = problem.residual(result.x)
        return dict(x=result.x, score=float(r@r), nfev=int(result.nfev), local_status=int(result.status),stop_reason=stop_reason)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return dict(x=x, score=1e30, nfev=0)

def signature(project, manifest, layout):


    p = deepcopy(project)
    algorithm={k:p['search'].get(k,DEFAULT_SEARCH.get(k)) for k in ('seed','starts','keep','short_nfev','max_nfev','local_time_limit_seconds')}
    for key in ('search','output_root','reference_result'): p.pop(key, None)
    p['reproducible_search_signature']=algorithm
    p['datasets'] = [r for r in p['datasets'] if r['training']]
    h = hashlib.sha256(json.dumps(p,sort_keys=True).encode())
    h.update(json.dumps(manifest,sort_keys=True).encode())
    h.update(json.dumps(dict(entries=layout.entries, fixed=layout.fixed),sort_keys=True).encode())
    for name in ('unified.py','project.py','run_control.py'): h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()

def search(problem, out, manifest, log=print, cancel=None):
    settings = problem.project['search']; layout = problem.layout
    fingerprint = signature(problem.project, manifest, layout)
    begin = time.monotonic();active_begin=active_time(cancel)
    deadline = active_begin+settings['time_limit_seconds'] if settings['time_limit_seconds'] else 0
    problem.run_control=cancel if isinstance(cancel,RunControl) else None
    pause_checkpoint(cancel)
    x = layout.initial.copy(); r = problem.residual(x)
    archive = [dict(x=x, score=float(r@r), nfev=1)]

    initial_candidates = []
    for candidate in problem.project.get('initial_candidates', []):
        unknown = set(candidate)-set(layout.entries)
        if unknown: raise ValueError(f'Initial candidate contains unknown parameters: {unknown}')
        values = layout.unpack(layout.initial)
        for name, value in candidate.items():
            _, low, high, _ = layout.entries[name]
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Initial candidate is outside its bounds: {name}={value}')
            if name in layout.fixed and value != layout.fixed[name]:
                raise ValueError(f'Initial candidate conflicts with a fixed parameter: {name}')
            values[name] = value
        vector = np.log([values[name] for name in layout.free])
        initial_candidates.append(vector)
        residual = problem.residual(vector)
        archive.append(dict(x=vector, score=float(residual@residual), nfev=1))
    done = 0; records = []; termination = 'budget_completed'
    resume = settings['resume_result']
    if resume:
        checkpoint = np.load(Path(resume)/'search_checkpoint.npz', allow_pickle=False)
        if str(checkpoint['signature']) != fingerprint: raise ValueError('Resume rejected because data, model, geometry, or sharing rules changed.')
        archive += [dict(x=a, score=float(b), nfev=0) for a,b in zip(checkpoint['x'],checkpoint['score'])]
        done = int(checkpoint['batches'])
    workers = settings['workers'] or min(4, max(1,(os.cpu_count() or 2)//2))
    workers = min(workers, settings['starts'])
    relay_stop=threading.Event();shared_cancel=None
    if workers>1:
        context=mp.get_context('spawn');shared_cancel=cancel if isinstance(cancel,RunControl) else context.Event()
        if cancel is not None and not isinstance(cancel,RunControl):
            def relay():
                while not relay_stop.wait(.1):
                    if cancel.is_set():shared_cancel.set();return
            threading.Thread(target=relay,daemon=True).start()
    try:
        pool = ProcessPoolExecutor(max_workers=workers,mp_context=context, initializer=_init_worker, initargs=(problem,shared_cancel)) if workers>1 else None
    except OSError as exc:
        log(f'Parallel workers are unavailable; using one process: {exc}')
        workers=1;pool=None
    if pool is None: _init_worker(problem,cancel)
    def stopped():
        pause_checkpoint(cancel)
        return (cancel is not None and cancel.is_set()) or (deadline and active_time(cancel)>deadline)
    def save(batch):
        chosen = sorted(archive,key=lambda a:a['score'])[:max(settings['keep']*2,4)]
        temp = out/'search_checkpoint.tmp'
        with temp.open('wb') as fh:
            np.savez_compressed(fh,x=np.asarray([a['x'] for a in chosen]),score=np.asarray([a['score'] for a in chosen]),batches=batch,signature=fingerprint)
        temp.replace(out/'search_checkpoint.npz')
    def run(xs, short, budget):
        results = []
        for start in range(0,len(xs),workers):
            if stopped(): break

            local_limit=settings['local_time_limit_seconds']
            wave_deadline=deadline
            if local_limit:wave_deadline=min(deadline,active_time(cancel)+local_limit) if deadline else active_time(cancel)+local_limit
            tasks = [(a,short,budget,wave_deadline) for a in xs[start:start+workers]]
            batch = list(pool.map(_solve_local,tasks)) if pool else list(map(_solve_local,tasks))
            archive.extend(batch); results.extend(batch); save(done)
            log(f"{'Screening' if short else 'Refinement'} {min(start+workers,len(xs))}/{len(xs)}; objective={min(a['score'] for a in archive):.6g}")
        return results
    try:
        save(done)
        for batch in range(done, done+settings['batches']):
            if stopped(): break
            if len(layout.free):
                sobol = qmc.Sobol(len(layout.free),scramble=True,seed=settings['seed']+batch*1009)
                u = sobol.random_base2(math.ceil(math.log2(settings['starts'])))[:settings['starts']]

                lo = np.maximum(layout.lower,layout.initial-math.log(1e3)); hi = np.minimum(layout.upper,layout.initial+math.log(1e3))
                starts = lo+u*(hi-lo); starts[3::4] = layout.lower+u[3::4]*(layout.upper-layout.lower)
                starts[0] = layout.initial if batch==0 else min(archive,key=lambda a:a['score'])['x']
                if batch == 0:
                    for i, vector in enumerate(initial_candidates[:len(starts)-1], 1):
                        starts[i] = vector
            else: starts = np.zeros((1,0))
            sampled = run(starts,True,settings['short_nfev'])
            ranked = sorted(sampled,key=lambda a:a['score']); selected = []
            for candidate in ranked:
                if not selected or min(np.linalg.norm(candidate['x']-v)/math.sqrt(max(len(v),1)) for v in selected)>.3:
                    selected.append(candidate['x'])
                if len(selected)>=settings['keep']: break
            selected += [min(archive,key=lambda a:a['score'])['x'], starts[-1]]
            refined = run(selected,False,settings['max_nfev'])
            completed = len(sampled)==len(starts) and len(refined)==len(selected)
            records.append(dict(batch=batch, completed=completed, seed=settings['seed']+batch*1009,
                                best=min((a['score'] for a in refined), default=None)))
            done = batch+1 if completed else batch; save(done)
            log(f'Exploration batch {batch+1}: completed={completed}; full-frequency score used.')
            if not completed: break
        if stopped(): termination = 'user_stopped' if cancel is not None and cancel.is_set() else 'time_budget_reached'
    finally:
        if pool: pool.shutdown(wait=True,cancel_futures=True)
        elif _LIMITER is not None: _LIMITER.restore_original_limits()
        relay_stop.set()
    best = min(archive,key=lambda a:a['score']); save(done)
    if best['score']>=1e25: raise RuntimeError('No valid network solution was obtained.')
    info = dict(termination=termination, signature=fingerprint, elapsed_seconds=time.monotonic()-begin,
                workers=workers, backend='CPU float64/complex128, sparse Jacobian, passive ladder recursion',
                global_optimum_proven=False, batches=records, objective=best['score'],
                local_step_time_cap_seconds=settings['local_time_limit_seconds'], candidates=len(archive), free_parameters=layout.free)
    info['local_time_capped_tasks']=sum(a.get('stop_reason')=='time_limit' for a in archive)
    info['active_elapsed_seconds']=active_time(cancel)-active_begin
    info['paused_seconds']=max(0.,info['elapsed_seconds']-info['active_elapsed_seconds'])
    info['local_user_stopped_tasks']=sum(a.get('stop_reason')=='user_stopped' for a in archive)
    info['local_evaluation_limited_tasks']=sum(a.get('local_status')==0 for a in archive)
    info['geometry_inactive_fixed_parameters']=getattr(layout,'geometry_inactive',[])
    write_json(out/'optimizer_summary.json',info)
    np.savez_compressed(out/'candidate_archive.npz',x=np.asarray([a['x'] for a in sorted(archive,key=lambda a:a['score'])[:20]]),
                        score=np.asarray([a['score'] for a in sorted(archive,key=lambda a:a['score'])[:20]]))
    return best['x'], info

def diagnostics(problem, x):
    base = problem.residual(x); rows = []; jac = []
    for i,key in enumerate(problem.layout.free):
        lo,hi = problem.layout.lower[i],problem.layout.upper[i]
        xp=x.copy(); xm=x.copy(); xp[i]=min(x[i]+1e-3,hi); xm[i]=max(x[i]-1e-3,lo)
        j=(problem.residual(xp)-problem.residual(xm))/(xp[i]-xm[i]); jac.append(j)
        rows.append(dict(parameter=key,value=float(np.exp(x[i])),
            lower=float(np.exp(lo)),upper=float(np.exp(hi)),
            lower_distance_decades=float((x[i]-lo)/math.log(10)),
            upper_distance_decades=float((hi-x[i])/math.log(10)),
            nearest_bound='lower' if x[i]-lo < hi-x[i] else 'upper',
            near_bound=bool(min(x[i]-lo,hi-x[i]) < .01*(hi-lo)),
            log_parameter_sensitivity=float(np.linalg.norm(j)),
            note='Local sensitivity is not a confidence interval; bounds are numerical domains'))
    if jac:
        s=np.linalg.svd(np.asarray(jac).T,compute_uv=False)
        weak=int(np.count_nonzero(s<max(s[0],1e-30)*1e-6))
    else: s=np.array([]); weak=0
    return rows,dict(singular_values=s.tolist(),weak_directions=weak,confidence_intervals_computed=False)

def export(project, problem, x, out, manifest, duplicates, info, log=print):
    from .project_results import export_project_results
    return export_project_results(project, problem, x, out, manifest, duplicates, info, log)

def execute_project(project, log=print, cancel=None):
    pause_checkpoint(cancel)
    project=validate_project(project)
    traces,manifest,duplicates=load_traces(project,True)
    if not traces: raise ValueError('Select at least one training dataset.')
    reference={}; ref_manifest=[]
    if project.get('reference_result'):
        ref=Path(project['reference_result'])
        raw=json.loads((ref/'parameters.json').read_text(encoding='utf-8'))
        if raw['model_id']!=MODEL_ID:
            raise ValueError('The reference must be generated by the current model.')
        reference=raw['values']
        if (ref/'input_manifest.json').exists(): ref_manifest=json.loads((ref/'input_manifest.json').read_text(encoding='utf-8'))
    mode=project.get('reference_mode','seed')
    overlap_suffixes=('/rc_au_pedot_area','/rct_overlap','/cdl_overlap','/warburg_overlap')
    if mode!='seed' and not reference:raise ValueError('A fixed-overlap run requires a current reference result.')
    if reference and mode!='seed' and {r['sha256'] for r in manifest}&{r['sha256'] for r in ref_manifest}:
        raise ValueError('Reference observations are already in the training set; use them only as initial values.')
    out=Path(project['output_root'])/datetime.now().strftime('unified_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'project.json',project); write_json(out/'input_manifest.json',manifest)
    baseline=deepcopy(project)
    if reference and mode=='fixed_overlap':
        overrides=baseline.setdefault('parameters',{})
        for key,value in reference.items():
            if key.endswith(overlap_suffixes) and key.split('/')[0] in {t.material for t in traces}:
                overrides[key]=dict(value=value,fixed=True)
    layout=Layout(baseline,traces,reference); problem=Problem(baseline,traces,layout)
    log(f'Network: {len(traces)} spectra, {len(layout.free)} free parameters; lumped bulk polarization={project["options"]["bulk_concentration_polarization"]}.')
    if duplicates: log(f'{len(duplicates)} duplicate spectra were excluded from the fitting weight.')
    x,info=search(problem,out,manifest,log,cancel)
    export(baseline,problem,x,out,manifest,duplicates,info,log)
    log(f'Completed: {out}')
    return out

def predict_locked(result_folder, output_root=None, compare=False, log=print):
    from .project_results import predict_result
    return predict_result(result_folder,output_root,compare,log)

def profile_parameter(result_folder, key, points=9, max_nfev=500):

    root=Path(result_folder); project=load_project(root/'project.json')
    traces,manifest,_=load_traces(project,True)
    original=json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    if manifest!=original: raise ValueError('Profile rejected because training data changed.')
    saved=json.loads((root/'parameters.json').read_text(encoding='utf-8'))
    layout=Layout(project,traces,saved['values'])
    if key not in layout.entries: raise ValueError('Unknown parameter.')
    _,lo,hi,_=layout.entries[key]; values=np.geomspace(lo,hi,points); rows=[]
    for v in values:
        local=deepcopy(project); local.setdefault('parameters',{})[key]=dict(value=float(v),lower=lo,upper=hi,fixed=True)
        ll=Layout(local,traces,saved['values']); pp=Problem(local,traces,ll)
        _init_worker(pp)
        try: r=_solve_local((ll.initial,False,max_nfev,time.monotonic()+45))
        finally:
            if _LIMITER is not None: _LIMITER.restore_original_limits()
        rows.append(dict(parameter=key,value=v,objective=r['score'],nfev=r['nfev']))
    out=root/('profile_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
    pd.DataFrame(rows).to_csv(out,index=False)
    return out

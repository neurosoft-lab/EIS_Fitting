from __future__ import annotations
import json,copy,os,platform,sys,traceback,shutil,uuid
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from .unified import MODEL_ID,SPECS,spectrum
from .project import DEFAULT_OPTIONS,DEFAULT_SEARCH,execute_project,validate_project,Layout,Trace,sha,write_json
from .project_results import metrics
from .run_control import pause_checkpoint

DISPLAY={'DCH':'DCH','ap-DCH':'ap-DCH','ICH':'ICH'}
MATERIALS=tuple(DISPLAY)
STUDIES={'Overlap':'1_Overlap','Length':'2_Length','Normalized interconnect area':'3_Normalized interconnect area','Width':'4_Width'}
STUDY_LABELS={'Overlap':'Overlap','Length':'Length','Normalized interconnect area':'Normalized interconnect area','Width':'Width'}
OVERLAP=('rc_au_pedot_area','rct_overlap','cdl_overlap','warburg_overlap')
PARAM_LABELS={
 'rho_i':'Ionic resistivity','rho_e':'PEDOT resistivity','rc_au_pedot_area':'Au–PEDOT contact resistivity',
 'rct_overlap':'Overlap coupling resistance','cdl_overlap':'Overlap double-layer capacitance','warburg_overlap':'Overlap Warburg coefficient',
 'rct_bulk':'Bulk coupling resistance','cdl_bulk':'Bulk double-layer capacitance','warburg_bulk':'Bulk Warburg coefficient',
 'rt_tip':'Tip transfer resistance','qt_tip':'Tip CPE coefficient','nt_tip':'Tip CPE exponent','warburg_tip':'Tip Warburg coefficient',
 'bulk_pol_r_ref':'Bulk polarization resistance','bulk_pol_tau_ref':'Bulk polarization time constant',
 'bulk_pol_geometry_exponent':'Bulk polarization geometry exponent','bulk_pol_alpha':'Bulk polarization exponent','Rs':'Series resistance'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp():return datetime.now().strftime('%Y%m%d_%H%M%S_%f')
def identify(study,material):return study+'_'+material.replace(' ','_')
def app_data_root(app):return Path(app).resolve()/'01_Data'
def group_path(root,study,material):return Path(root)/STUDIES[study]/DISPLAY[material]
def fitting_root(app,study,material):return group_path(Path(app)/'03_Bin'/'Results',study,material)/'Fitting'

def device_data_dirs(app,study,material,data_root=None,require_data=True):

    root=Path(data_root) if data_root else group_path(app_data_root(app),study,material)
    direct=root/'Rawdata_txt'
    if direct.is_dir():return [(root.name,direct)]
    found=[]
    for index in range(1,5):
        folder=root/f'Device{index}'/'Rawdata_txt'
        if require_data and not folder.is_dir():raise ValueError(f'Missing independent-device data folder: {folder}')
        found.append((f'Device{index}',folder))
    return found

def _atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(str(text),encoding='utf-8')
    os.replace(tmp,path)

def resolve_result_folder(folder):

    current=Path(folder).resolve();visited=set()
    for _ in range(8):
        if current in visited:raise ValueError('Circular result pointer: '+str(current))
        visited.add(current)
        if (current/'parameters.json').is_file() and (current/'project.json').is_file():return current
        manifest=current/'run_manifest.json'
        if manifest.is_file():
            selected=read(manifest).get('selected_result','')
            if selected:
                candidate=Path(selected)
                current=(candidate if candidate.is_absolute() else current/candidate).resolve();continue
        pointer=None
        for name in ('SELECTED_RESULT.txt','LATEST_RESULT.txt','LATEST.txt'):
            item=current/name
            if item.is_file() and item.read_text(encoding='utf-8-sig').strip():
                pointer=item.read_text(encoding='utf-8-sig').strip();break
        if pointer:
            candidate=Path(pointer)
            current=(candidate if candidate.is_absolute() else current/candidate).resolve();continue
        raise ValueError('The selected folder is not a complete result and has no valid result pointer: '+str(current))
    raise ValueError('Result pointer nesting is too deep: '+str(folder))

def latest_study_result(app,study,material):
    root=fitting_root(app,study,material)
    try:return resolve_result_folder(root)
    except (ValueError,OSError,KeyError,json.JSONDecodeError):return None

def current_model_result(path):
    path=Path(path)
    try:
        project=read(path/'project.json')
        datasets=[r for r in project.get('datasets',[]) if r.get('training',True)]
        valid_devices={r.get('device') for r in datasets}=={f'Device{i}' for i in range(1,5)}
        return path if read(path/'parameters.json').get('model_id')==MODEL_ID and valid_devices else None
    except (OSError,KeyError,json.JSONDecodeError):return None

def default_reference(app,material):
    fitting=fitting_root(app,'Overlap',material)
    selected=fitting/'SELECTED_REFERENCE.txt'
    if selected.is_file() and selected.read_text(encoding='utf-8-sig').strip():
        p=Path(selected.read_text(encoding='utf-8-sig').strip())
        if not p.is_absolute():p=fitting/p
        try:
            resolved=current_model_result(resolve_result_folder(p))
            if resolved:return str(resolved)
        except (ValueError,OSError,KeyError,json.JSONDecodeError):pass
    for name in ('selected.json','latest.json'):
        local=Path(app)/'02_Reference'/DISPLAY[material]/name
        if not local.exists():continue
        p=Path(read(local)['result'])
        if not p.is_absolute():p=local.parent/p
        try:
            resolved=current_model_result(resolve_result_folder(p))
            if resolved:return str(resolved)
        except (ValueError,OSError,KeyError,json.JSONDecodeError):pass
    candidate=latest_study_result(app,'Overlap',material)
    candidate=current_model_result(candidate) if candidate else None
    return str(candidate) if candidate else ''

def make_project(app,study,material,data_root=None,require_data=True):
    catalog=read(Path(app)/'03_Bin/Templates/catalog.json')
    template=copy.deepcopy(next(r for r in catalog['datasets'] if r['study']==study and r['material']==material))
    datasets=[]
    for device,folder in device_data_dirs(app,study,material,data_root,require_data):
        row=copy.deepcopy(template)
        row.update(id=f'{template["id"]}_{device}',batch=f'{template["id"]}_{device}',
                   training=True,replicate=1,data_dir=str(folder.resolve()),device=device)
        datasets.append(row)
    options=copy.deepcopy(DEFAULT_OPTIONS);options['mesh_step_um']=25. if study=='Width' else 100.


    options['bulk_concentration_polarization']=(
        study in ('Length','Normalized interconnect area') and material in ('ICH','ap-DCH'))
    parameters={}
    return dict(model_id=MODEL_ID,output_root=str(fitting_root(app,study,material)),datasets=datasets,options=options,
                search=copy.deepcopy(DEFAULT_SEARCH),parameters=parameters,reference_mode='seed',reference_result='',initial_candidates=[])

def geometry_layout(project):
    p=validate_project(project)
    traces=[Trace(r['id'],r['material'],r['batch'],ch,1,g,np.array([1.,10.]),np.ones(2,complex),'geometry-only','')
            for r in p['datasets'] for ch,g in enumerate(r['geometry'],1)]
    return Layout(p,traces)

def read_bundle(folder):
    root=resolve_result_folder(folder);p=read(root/'project.json');saved=read(root/'parameters.json')
    models={p.get('model_id'),saved.get('model_id')}
    if models!={MODEL_ID}:raise ValueError('The selected result was not generated by the current model.')
    lock=root/'prediction_lock.json'
    if lock.exists():
        info=read(lock)
        if info.get('model_code_sha256')!=sha(Path(__file__).with_name('unified.py')):raise ValueError('The result model signature does not match.')
        for name,digest in info.get('files',{}).items():
            item=(root/name).resolve()
            if not item.is_relative_to(root):raise ValueError('The result contains an out-of-scope path.')
            if sha(item)!=digest:raise ValueError('The result package was modified: '+name)
    rows=[copy.deepcopy(r) for r in p['datasets'] if r.get('training',True)]
    if not rows:raise ValueError('The result has no training datasets.')
    signatures={(r['study'],r['material']) for r in rows}
    if len(signatures)!=1:raise ValueError('Select one complete single-study, single-material result package.')
    row=rows[0];curves=[]
    for dataset in rows:
      for ch in range(1,9):
        rep=int(dataset.get('replicate',1))
        source=root/'datasets'/dataset['id']/'fitted_curves'/f'ch{ch}-{rep}_fitted.csv'
        if not source.exists():raise ValueError(f'Saved observations are missing for {dataset["id"]} CH{ch}.')
        table=pd.read_csv(source).sort_values('Frequency_Hz')
        names=['Frequency_Hz','Zobs_Re_Ohm','Zobs_minusIm_Ohm','Zfit_Re_Ohm','Zfit_minusIm_Ohm']
        if not set(names)<=set(table) or not np.isfinite(table[names]).all().all():raise ValueError('Observation columns are missing or non-finite.')
        f=table.Frequency_Hz.to_numpy();obs=table.Zobs_Re_Ohm.to_numpy()-1j*table.Zobs_minusIm_Ohm.to_numpy()
        if np.any(f<=0) or np.any(abs(obs)==0) or len(np.unique(f))!=len(f):raise ValueError('Invalid frequency or zero impedance.')
        fit=table.Zfit_Re_Ohm.to_numpy()-1j*table.Zfit_minusIm_Ohm.to_numpy()
        curves.append(dict(dataset=dataset['id'],device=dataset.get('device',dataset['id']),channel=ch,replicate=rep,
                           frequency=f,obs=obs,fit=fit,source=str(source),sha256=sha(source)))
        before=root/'before_adjustment'/dataset['id']/f'ch{ch}-{rep}_fitted.csv'
        if before.exists():
            previous=pd.read_csv(before).sort_values('Frequency_Hz')
            if not np.array_equal(previous.Frequency_Hz.to_numpy(),f):raise ValueError('Stored comparison frequencies do not match.')
            curves[-1]['baseline']=previous.Zfit_Re_Ohm.to_numpy()-1j*previous.Zfit_minusIm_Ohm.to_numpy()
    fixed_table=root/'global_fit_parameters.csv'
    if fixed_table.exists():
        ft=pd.read_csv(fixed_table);fixed_keys=set(ft.loc[ft.Fixed.astype(bool),'Parameter'])
    else:
        fixed_keys={k for k,v in p.get('parameters',{}).items() if v.get('fixed',False)}
        test=copy.deepcopy(p);test['parameters']={}
        fixed_keys.update(geometry_layout(test).fixed)
    return dict(root=root,project=p,row=row,rows=rows,saved=saved,curves=curves,fixed_keys=fixed_keys)

def replay_bundle(bundle):
    p=bundle['project'];r=bundle['row'];v=bundle['saved']['values'];mat=bundle['saved']['material_values'][r['material']]
    rows={row['id']:row for row in bundle.get('rows',[r])}
    errors=[]
    for curve in bundle['curves']:
        dataset=rows[curve.get('dataset',r['id'])];g=dataset['geometry'][curve['channel']-1]
        rs=v[f'measurement/{dataset["batch"]}/Rs']
        z,_=spectrum(r['material'],mat,g,curve['frequency'],p['options'],rs)
        errors.append(float(np.max(abs(z-curve['fit'])/np.maximum(abs(curve['fit']),1e-30))))
    return max(errors)

def snapshot_bundle(bundle,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);manifest=[];folders={}
    for c in bundle['curves']:
        dataset=c.get('dataset',bundle['row']['id'])
        folder=dest/dataset;folder.mkdir(exist_ok=True);folders[dataset]=folder
        out=folder/f'ch{c["channel"]}-{c.get("replicate",1)}.txt'
        pd.DataFrame({'Frequency_Hz':c['frequency'],'Re_Ohm':c['obs'].real,'minusIm_Ohm':-c['obs'].imag}).to_csv(out,sep='\t',index=False)
        manifest.append(dict(dataset=dataset,channel=c['channel'],source=c['source'],source_sha256=c['sha256'],snapshot_sha256=sha(out)))
    write_json(dest/'snapshot_provenance.json',dict(kind='saved_result_observations_NOT_current_Device',source_result=str(bundle['root']),files=manifest))
    return folders

def refit_project(bundle,output_root,search=None):
    p=copy.deepcopy(bundle['project']);p['datasets']=[copy.deepcopy(r) for r in bundle.get('rows',[bundle['row']])]
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    snaps=snapshot_bundle(bundle,root/('_saved_observations_'+stamp()))
    for row in p['datasets']:row.update(data_dir=str(snaps[row['id']]),training=True,replicate=1)
    p['output_root']=str(root);p['reference_result']='';p['reference_mode']='seed';p['initial_candidates']=[]
    p['search']=copy.deepcopy(DEFAULT_SEARCH)|(search or {})
    for key,value in bundle['saved']['values'].items():
        item=p.setdefault('parameters',{}).setdefault(key,{})
        item['value']=value
        item['fixed']=key in bundle['fixed_keys']
    p['seed_provenance']=dict(source=str(bundle['root']),type='explicit_previous_fit_initial_guess',observations='saved_result_columns')
    return validate_project(p)

def runtime_info():
    import scipy
    return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
                solver_sha256=sha(Path(__file__).with_name('unified.py')),workflow_sha256=sha(Path(__file__).with_name('project.py')))

def result_quality(root):
    root=Path(root);m=pd.read_csv(root/'channel_metrics.csv');sens=pd.read_csv(root/'parameter_sensitivity.csv')
    mesh=float(pd.read_csv(root/'mesh_convergence.csv').max_relative_change_percent.max())
    return dict(RMS_percent=float(np.sqrt(np.mean(m.relative_complex_RMS_percent**2))),mesh_max_percent=mesh,
                mesh_passed=mesh<1.,weak_directions=read(root/'identifiability.json')['weak_directions'],
                near_bounds=sens.loc[sens.near_bound.astype(bool),'parameter'].tolist())

def _run_manifest(project,parent,policy,reference,factor,status='running',**extra):
    row=project['datasets'][0]
    payload=dict(schema='eis-fitting-run',status=status,run_folder=parent.name,
        study=row['study'],study_folder=STUDIES[row['study']],material=row['material'],material_folder=DISPLAY[row['material']],
        dataset_ids=[r['id'] for r in project['datasets']],data_dirs=[r.get('data_dir','') for r in project['datasets']],
        replicate=row.get('replicate',1),independent_devices=len(project['datasets']),channels='ch1-1 through ch8-1 per device',
        model_id=project.get('model_id',MODEL_ID),reference_policy=policy,reference_result=str(reference or ''),
        reference_factor=float(factor),options=copy.deepcopy(project.get('options',{})),search=copy.deepcopy(project.get('search',{})),
        started_at=extra.pop('started_at',datetime.now().isoformat(timespec='seconds')))
    payload.update(extra);write_json(parent/'run_manifest.json',payload);return payload

def _finish_run(parent,project,manifest,out,stages,cancel,log):
    interrupted=bool(cancel and cancel.is_set())
    status='interrupted' if interrupted else 'completed'
    manifest.update(status=status,completed_at=datetime.now().isoformat(timespec='seconds'),
        selected_result=os.path.relpath(Path(out).resolve(),parent.resolve()),
        stages=[dict(path=os.path.relpath(Path(s).resolve(),parent.resolve()),**result_quality(s)) for s in stages],
        interrupted=interrupted)
    write_json(parent/'run_manifest.json',manifest)
    _atomic_text(parent/'SELECTED_RESULT.txt',manifest['selected_result']+'\n')
    explanation=[f"Status: {status}",f"Study: {manifest['study']}",f"Material: {DISPLAY[manifest['material']]}",
        f"Started: {manifest['started_at']}",f"Completed: {manifest['completed_at']}",f"Result: {manifest['selected_result']}",
        f"Reference workflow: {manifest['reference_policy']}",f"Reference: {manifest['reference_result'] or 'none'}",
        'LATEST identifies the most recent completed run; it is not a physical model-selection claim.']
    _atomic_text(parent/'RUN_STATUS.txt','\n'.join(explanation)+'\n')
    if status=='completed':
        fitting=parent.parent
        _atomic_text(fitting/'LATEST.txt',parent.name+'\n')
        _atomic_text(fitting/'LATEST_RESULT.txt',os.path.relpath(Path(out).resolve(),fitting.resolve())+'\n')
        log('Updated completed-result pointer: '+str(fitting/'LATEST.txt'))

def run_study(project,policy='auto',reference='',factor=3.,log=print,cancel=None,saved_bundle=None):
    pause_checkpoint(cancel)
    p=validate_project(project)
    signatures={(r['study'],r['material']) for r in p['datasets']}
    if len(signatures)!=1:raise ValueError('One fit must contain one study and one material across independent devices.')
    row=p['datasets'][0]
    if policy=='auto':
        policy='none' if row['study']=='Overlap' else 'staged'
    if policy not in ('none','staged'):raise ValueError('Invalid reference workflow.')
    if policy=='staged' and not reference:raise ValueError('Fit the matching Overlap study before this study.')
    p['reference_mode']='seed';p['reference_result']=''

    p['output_root']=str(Path(p['output_root']).resolve())
    parent=Path(p['output_root'])/('run_'+stamp());parent.mkdir(parents=True,exist_ok=False)
    manifest=_run_manifest(p,parent,policy,reference,factor)
    _atomic_text(parent/'RUN_STATUS.txt','Status: running\nThis run is incomplete.\n')
    write_json(parent/'runtime.json',runtime_info())
    try:
        if saved_bundle is not None:
            snaps=snapshot_bundle(saved_bundle,parent/'saved_observations')
            for row in p['datasets']:row.update(data_dir=str(snaps[row['id']]),training=True,replicate=1)
            manifest['saved_observation_source']=str(saved_bundle['root'])
            write_json(parent/'run_manifest.json',manifest)
        if policy=='none':
            p['output_root']=str(parent);out=execute_project(p,log,cancel)
            stages=[out]
        else:
            ref=read_bundle(reference);row=p['datasets'][0]
            if ref['row']['material']!=row['material'] or ref['row']['study']!='Overlap':raise ValueError('The reference must be a complete matching-material Overlap result.')
            if {r.get('device') for r in ref['rows']}!={f'Device{i}' for i in range(1,5)}:
                raise ValueError('The Overlap reference must contain Device1 through Device4 from the multi-device workflow.')
            rq=result_quality(ref['root'])
            log(f"Reference: RMS {rq['RMS_percent']:.3g}%, mesh change {rq['mesh_max_percent']:.3g}%, near-bound parameters {len(rq['near_bounds'])}.")
            if not rq['mesh_passed']:raise ValueError('The Overlap reference did not pass mesh convergence.')
            if not np.isfinite(factor) or factor<=1:raise ValueError('Reference release factor must exceed 1.')
            p['reference_result']=str(ref['root']);p['reference_mode']='fixed_overlap'
            p['output_root']=str(parent/'01_reference_fixed')
            layout=geometry_layout(p);rv=ref['saved']['material_values'][row['material']]
            for key in layout.free:
                bare=key.split('/')[-1]
                if bare in rv and key not in p['parameters']:
                    if layout.entries[key][1]<=rv[bare]<=layout.entries[key][2]:p['parameters'][key]={'value':rv[bare]}
            first=execute_project(p,log,cancel);stages=[first];out=first
            pause_checkpoint(cancel)
            if not(cancel and cancel.is_set()):
                released=copy.deepcopy(p);released['reference_mode']='seed'
                values=read(first/'parameters.json')['values'];fixed_table=pd.read_csv(first/'global_fit_parameters.csv')
                for key,value in values.items():
                    released['parameters'].setdefault(key,{})['value']=value
                    if bool(fixed_table.loc[fixed_table.Parameter==key,'Fixed'].iloc[0]):released['parameters'][key]['fixed']=True
                calibration={}
                for bare in OVERLAP:
                    key=row['material']+'/'+bare
                    if key not in values:continue
                    value=rv[bare];_,lo,hi,_=SPECS[bare];original=project.get('parameters',{}).get(key,{})
                    lo=max(lo,value/factor,original.get('lower',lo));hi=min(hi,value*factor,original.get('upper',hi))
                    if not lo<hi or not lo<=value<=hi:raise ValueError('Reference bounds conflict with the parameter table: '+key)
                    released['parameters'][key]=dict(value=value,lower=lo,upper=hi,fixed=False)
                    calibration[key]=dict(reference=value,lower=lo,upper=hi)
                released['reference_calibration']=calibration;released['output_root']=str(parent/'02_reference_calibrated')
                released['search']['resume_result']=''
                second=execute_project(released,log,cancel);stages.append(second)
                qa,qb=result_quality(first),result_quality(second)
                if qa['mesh_passed']!=qb['mesh_passed']:out=first if qa['mesh_passed'] else second
                else:out=min(stages,key=lambda r:read(r/'optimizer_summary.json')['objective'])
        record=dict(selected=str(out),stages=[dict(path=str(s),**result_quality(s)) for s in stages],policy=policy,
                    selection='Prefer mesh pass if only one passes; otherwise same-objective lower error. Not physical validation.',
                    interrupted=bool(cancel and cancel.is_set()))
        write_json(parent/'selection.json',record)
        _finish_run(parent,p,manifest,out,stages,cancel,log)
        log('Selected result: '+str(out));return out
    except Exception as exc:
        manifest.update(status='failed',failed_at=datetime.now().isoformat(timespec='seconds'),error=str(exc),traceback=traceback.format_exc())
        write_json(parent/'run_manifest.json',manifest)
        _atomic_text(parent/'RUN_STATUS.txt','Status: failed\nError: '+str(exc)+'\n')
        raise

def resume_study(project,log=print,cancel=None):

    pause_checkpoint(cancel);p=validate_project(project)
    if len({(r['study'],r['material']) for r in p['datasets']})!=1:raise ValueError('Checkpoint resume accepts one study and material.')
    parent=Path(p['output_root'])/('run_'+stamp());parent.mkdir(parents=True,exist_ok=False)
    manifest=_run_manifest(p,parent,'resume_checkpoint','',3.,resume_source=p['search'].get('resume_result',''))
    _atomic_text(parent/'RUN_STATUS.txt','Status: running\nCheckpoint continuation is incomplete.\n')
    write_json(parent/'runtime.json',runtime_info())
    try:
        p['output_root']=str(parent/'resume_checkpoint')
        out=execute_project(p,log,cancel);stages=[out]
        write_json(parent/'selection.json',dict(selected=str(out),stages=[dict(path=str(out),**result_quality(out))],
            policy='resume_checkpoint',selection='Single resumed result',interrupted=bool(cancel and cancel.is_set())))
        _finish_run(parent,p,manifest,out,stages,cancel,log)
        log('Resumed result: '+str(out));return out
    except Exception as exc:
        manifest.update(status='failed',failed_at=datetime.now().isoformat(timespec='seconds'),error=str(exc),traceback=traceback.format_exc())
        write_json(parent/'run_manifest.json',manifest)
        _atomic_text(parent/'RUN_STATUS.txt','Status: failed\nError: '+str(exc)+'\n')
        raise

def register_reference(app,result):
    b=read_bundle(result)
    if b['row']['study']!='Overlap':raise ValueError('Only an Overlap result can be registered as a reference.')
    folder=Path(app)/'02_Reference'/DISPLAY[b['row']['material']]
    folder.mkdir(parents=True,exist_ok=True)
    current=folder/'current'
    if b['root'].resolve()==current.resolve():return current
    token=uuid.uuid4().hex
    pending=folder/('pending_'+token);previous=folder/('previous_'+token)
    try:
        shutil.copytree(b['root'],pending)
        read_bundle(pending)
        if current.exists():current.rename(previous)
        try:pending.rename(current)
        except Exception:
            if previous.exists():previous.rename(current)
            raise
    finally:
        if pending.exists():shutil.rmtree(pending)
    if previous.exists():shutil.rmtree(previous)
    write_json(folder/'latest.json',dict(result='current'))
    return current

def register_selected_reference(app,result):
    return register_reference(app,result)


def discover_bundles(app):
    app=Path(app);candidates=[]
    explicit=app/'02_Reference'
    if explicit.is_dir():candidates+=list(explicit.rglob('parameters.json'))
    for study in STUDIES:
        for mat in MATERIALS:
            fitting=fitting_root(app,study,mat)
            try:candidates.append(resolve_result_folder(fitting)/'parameters.json')
            except (ValueError,OSError,KeyError,json.JSONDecodeError):pass
    found=[];seen=set();warnings=[];explicit_ids=set()
    for p in candidates:
        try:
            b=read_bundle(p.parent)
            is_explicit=p.resolve().is_relative_to(explicit.resolve())
            if not is_explicit and b['row']['id'] in explicit_ids:continue
            if is_explicit:explicit_ids.add(b['row']['id'])
            fingerprint=(b['row']['id'],sha(p),tuple(c['sha256'] for c in b['curves']))
            if fingerprint in seen:continue
            seen.add(fingerprint);found.append(b)
        except (ValueError,OSError,KeyError) as exc:warnings.append(str(p.parent)+': '+str(exc))
    return found,warnings

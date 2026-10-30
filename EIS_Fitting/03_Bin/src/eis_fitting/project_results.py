from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import hashlib
import numpy as np
import pandas as pd
from .unified import MODEL_ID, spectrum, prepare_mesh
from .run_control import pause_checkpoint

def metrics(pred, obs):
    rel=abs(pred-obs)/abs(obs)
    return dict(relative_complex_RMS_percent=float(100*np.sqrt(np.mean(rel**2))),
                magnitude_log10_RMS=float(np.sqrt(np.mean(np.log10(abs(pred)/abs(obs))**2))),
                phase_RMSE_deg=float(np.sqrt(np.mean(np.angle(pred/obs,deg=True)**2))))

def frame(f,z,obs=None,parts=None):
    t=pd.DataFrame(dict(Frequency_Hz=f,Zfit_Re_Ohm=z.real,Zfit_minusIm_Ohm=-z.imag,
                        Abs_Zfit_Ohm=abs(z),Phase_fit_deg=np.angle(z,deg=True)))
    if obs is not None:
        t['Zobs_Re_Ohm']=obs.real;t['Zobs_minusIm_Ohm']=-obs.imag
        t['Abs_Zobs_Ohm']=abs(obs);t['Phase_obs_deg']=np.angle(obs,deg=True)
    for key,value in (parts or {}).items():
        if value is not None: t[key+'_Re_Ohm']=value.real;t[key+'_minusIm_Ohm']=-value.imag
    return t

def plot_group(folder, curves, title, prediction=False, control=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'Arial','axes.unicode_minus':False,'savefig.dpi':140})
    folder.mkdir(parents=True,exist_ok=True)
    individual=folder/'channel_plots';individual.mkdir(exist_ok=True)
    fb,ab=plt.subplots(4,4,figsize=(18,13),layout='constrained')
    fn,an=plt.subplots(2,4,figsize=(18,9),layout='constrained')
    label='Locked prediction' if prediction else 'Model fit'
    def bode(am,ap,f,z,obs):
        if obs is not None:
            am.plot(f,abs(obs),'o',ms=2.5,color='#0072B2');ap.plot(f,np.angle(obs,deg=True),'o',ms=2.5,color='#0072B2')
        am.plot(f,abs(z),color='#D55E00');ap.plot(f,np.angle(z,deg=True),color='#D55E00')
        am.set(xscale='log',yscale='log',ylabel='|Z| / Ω');ap.set(xscale='log',ylabel='Phase / °',xlabel='f / Hz')
        for ax in (am,ap):ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    def nyq(ax,z,obs):
        maximum=max(abs(z).max(),abs(obs).max() if obs is not None else 0)
        unit,scale=('MΩ',1e6) if maximum>1e6 else ('kΩ',1e3)
        if obs is not None:ax.plot(obs.real/scale,-obs.imag/scale,'o',ms=2.5,color='#0072B2')
        ax.plot(z.real/scale,-z.imag/scale,color='#D55E00');ax.set(xlabel=f"Z′ / {unit}",ylabel=f"−Z″ / {unit}")
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))

    for ch in range(1,9):
        pause_checkpoint(control)
        choices=[c for c in curves if c[0]==ch]
        row,col=2*((ch-1)//4),(ch-1)%4; ax=an.flat[ch-1]
        if not choices:
            ab[row,col].set_title(f'CH{ch} unavailable');ax.set_title(f'CH{ch} unavailable');continue
        _,rep,f,z,obs=sorted(choices,key=lambda a:a[1])[0]
        bode(ab[row,col],ab[row+1,col],f,z,obs);ab[row,col].set_title(f'CH{ch} / rep{rep}')
        nyq(ax,z,obs);ax.set_title(f'CH{ch}')
        fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,layout='constrained');bode(*axes,f,z,obs)
        fig.suptitle(f'{title} CH{ch} | {label}\nBlue: observed; orange: model');fig.savefig(individual/f'CH{ch:02}_Bode.png');plt.close(fig)
        fig,one=plt.subplots(figsize=(7,6),layout='constrained');nyq(one,z,obs)
        fig.suptitle(f'{title} CH{ch} | {label}');fig.savefig(individual/f'CH{ch:02}_Nyquist.png');plt.close(fig)
    fb.suptitle(title+f' | {label} Bode; blue: observed, orange: model')
    fn.suptitle(title+f' | {label} Nyquist; blue: observed, orange: model')
    for fig,name in ((fb,'bode_8ch'),(fn,'nyquist_8ch')):
        pause_checkpoint(control)
        fig.savefig(folder/(name+'.png'));fig.savefig(folder/(name+'.svg'));plt.close(fig)

def export_multidevice_summary(out, groups, control=None):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets=[]
    for curves in groups.values():
        if curves:datasets.append(curves)
    all_curves=[c for group in datasets for c in group]
    devices=sorted({dataset for dataset,curves in groups.items() if curves})
    if len(devices)<2:return
    rows=[];fig,axes=plt.subplots(4,4,figsize=(18,13),layout='constrained')
    for ch in range(1,9):
        pause_checkpoint(control)
        choices=[]
        for dataset in devices:
            matches=[c for c in groups[dataset] if c[0]==ch]
            if matches:choices.append(sorted(matches,key=lambda a:a[1])[0])
        if len(choices)<2:continue
        f=choices[0][2]
        if any(not np.array_equal(c[2],f) for c in choices):
            raise ValueError(f'Device frequency grids differ for CH{ch}; aggregate statistics were not interpolated.')
        fit=np.vstack([abs(c[3]) for c in choices]);obs=np.vstack([abs(c[4]) for c in choices])
        fit_phase=np.vstack([np.angle(c[3],deg=True) for c in choices]);obs_phase=np.vstack([np.angle(c[4],deg=True) for c in choices])
        om,os=obs.mean(0),obs.std(0,ddof=1);fm,fs=fit.mean(0),fit.std(0,ddof=1)
        op,ops=obs_phase.mean(0),obs_phase.std(0,ddof=1);fp,fps=fit_phase.mean(0),fit_phase.std(0,ddof=1)
        for i,hz in enumerate(f):rows.append(dict(Channel=ch,Frequency_Hz=hz,Independent_devices=len(choices),
            Obs_absZ_mean_Ohm=om[i],Obs_absZ_SD_Ohm=os[i],Fit_absZ_mean_Ohm=fm[i],Fit_absZ_SD_Ohm=fs[i],
            Obs_phase_mean_deg=op[i],Obs_phase_SD_deg=ops[i],Fit_phase_mean_deg=fp[i],Fit_phase_SD_deg=fps[i]))
        row,col=2*((ch-1)//4),(ch-1)%4;am,ap=axes[row,col],axes[row+1,col]
        am.loglog(f,om,'o',ms=2.5,color='#0072B2',label='Observed mean')
        am.fill_between(f,np.maximum(om-os,np.finfo(float).tiny),om+os,color='#0072B2',alpha=.18,label='Observed s.d.')
        am.loglog(f,fm,color='#D55E00',label='Joint model mean')
        am.fill_between(f,np.maximum(fm-fs,np.finfo(float).tiny),fm+fs,color='#D55E00',alpha=.12,label='Model s.d.')
        ap.semilogx(f,op,'o',ms=2.5,color='#0072B2');ap.fill_between(f,op-ops,op+ops,color='#0072B2',alpha=.18)
        ap.semilogx(f,fp,color='#D55E00');ap.fill_between(f,fp-fps,fp+fps,color='#D55E00',alpha=.12)
        am.set(ylabel='|Z| / Ω',title=f'CH{ch}');ap.set(ylabel='Phase / °',xlabel='f / Hz')
        for ax in (am,ap):ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Independent-device summary (mean ± s.d., n={len(devices)} devices)')
    pd.DataFrame(rows).to_csv(out/'origin_multidevice_mean_sd.csv',index=False)
    fig.savefig(out/'multidevice_bode_mean_sd.png');fig.savefig(out/'multidevice_bode_mean_sd.svg');plt.close(fig)

def export_project_results(project, problem, x, out, manifest, duplicates, info, log):
    from .project import write_json, timestamp, sha, diagnostics
    control=getattr(problem,'run_control',None);pause_checkpoint(control)
    values=problem.layout.unpack(x)
    materials={mat:problem.layout.material(values,mat) for mat in problem.layout.materials}
    write_json(out/'project.json',project)
    write_json(out/'parameters.json',dict(model_id=MODEL_ID,values=values,material_values=materials,
        interpretation='SI effective material coefficients; fitted values do not establish unique mechanisms'))
    write_json(out/'input_manifest.json',manifest);write_json(out/'duplicates_excluded.json',duplicates)
    rows=[]
    for key,(initial,lo,hi,unit) in problem.layout.entries.items():
        rows.append(dict(Parameter=key,Value=values[key],Unit=unit,Lower=lo,Upper=hi,
                         Fixed=key in problem.layout.fixed,Sharing='independent device' if key.startswith('measurement/') else 'material across devices and geometries',
                         Estimate='joint point estimate',Uncertainty='not estimated; use device-level resampling'))
    pd.DataFrame(rows).to_csv(out/'global_fit_parameters.csv',index=False,encoding='utf-8-sig')
    groups={}; errors=[]; geometry=[]; meshcheck=[]; profile_audit=[]
    by_id={d['id']:d for d in project['datasets']}
    for tr in problem.traces:
        pause_checkpoint(control)
        mat=materials[tr.material];rs=values[f'measurement/{tr.batch}/Rs']
        z,parts=spectrum(tr.material,mat,tr.geometry,tr.frequency,project['options'],rs)
        folder=out/'datasets'/tr.dataset;folder.mkdir(parents=True,exist_ok=True)
        curves=folder/'fitted_curves';curves.mkdir(exist_ok=True)
        table=frame(tr.frequency,z,tr.z,parts)
        table.to_csv(curves/f'ch{tr.channel}-{tr.replicate}_fitted.csv',index=False)
        table.to_csv(curves/f'ch{tr.channel}-{tr.replicate}_fitted.txt',sep='\t',index=False)
        groups.setdefault(tr.dataset,[]).append((tr.channel,tr.replicate,tr.frequency,z,tr.z))
        row=dict(dataset=tr.dataset,material=tr.material,channel=tr.channel,replicate=tr.replicate,**metrics(z,tr.z))
        for name,mask in [('low',tr.frequency<10),('mid',(tr.frequency>=10)&(tr.frequency<1000)),('high',tr.frequency>=1000)]:
            if mask.any():row[name+'_relative_RMS_percent']=metrics(z[mask],tr.z[mask])['relative_complex_RMS_percent']
        errors.append(row)
        mesh=prepare_mesh(tr.geometry,project['options']['mesh_step_um'])
        geometry.append(dict(dataset=tr.dataset,channel=tr.channel,replicate=tr.replicate,
            overlap_intervals=mesh.overlap_edges,total_intervals=len(mesh.axial),
            total_overlap_area_m2=float(mesh.overlap_area.sum()),total_bulk_volume_m3=float(mesh.bulk_volume.sum()),tip_area_m2=mesh.tip_area,
            physical_bulk_axial_integral_m_inv=float(mesh.axial[mesh.overlap_edges:].sum()),
            concentration_polarization_enabled=bool(project['options'].get('bulk_concentration_polarization',False)),
            concentration_polarization_model=('lumped_bulk' if project['options'].get('bulk_concentration_polarization',False) else 'none'),
            mesh_note='All geometry fields use the physical mesh without correction factors.'))
        fine=project['options']|{'mesh_step_um':project['options']['mesh_step_um']/2}
        zz,_=spectrum(tr.material,mat,tr.geometry,tr.frequency,fine,rs)
        meshcheck.append(dict(dataset=tr.dataset,channel=tr.channel,replicate=tr.replicate,
            max_relative_change_percent=float(100*np.max(abs(z-zz)/abs(z)))))
        from .network_diagnostics import network_profile
        frequencies=tr.frequency[np.unique([0,int(np.argmin(abs(np.log(tr.frequency/1000)))),len(tr.frequency)-1])]
        try:
            profiles,nodal_error=network_profile(tr.material,mat,tr.geometry,frequencies,project['options'])
            pd.DataFrame(profiles).to_csv(folder/f'ch{tr.channel}-{tr.replicate}_network_profile.csv',index=False)
            profile_audit.append(dict(dataset=tr.dataset,channel=tr.channel,status='verified',error=nodal_error))
        except (FloatingPointError,ValueError,np.linalg.LinAlgError) as exc:
            profile_audit.append(dict(dataset=tr.dataset,channel=tr.channel,status='not_exported',reason=str(exc)))
    for key,curves in groups.items():
        folder=out/'datasets'/key

        blocks=[]
        for ch,rep,f,z,obs in curves:
            block=frame(f,z,obs);block.columns=[f'CH{ch}_rep{rep}_{col}' for col in block.columns];blocks.append(block.reset_index(drop=True))
        pd.concat(blocks,axis=1).to_csv(folder/'origin_overlay_all_repeats.csv',index=False)
        primary=min(c[1] for c in curves)
        for rep in sorted({c[1] for c in curves}):
            bode=None
            for ch,repeat,f,z,obs in sorted(curves,key=lambda a:a[0]):
                if repeat!=rep:continue
                block=pd.DataFrame({'Frequency_Hz':f,f'CH{ch}_obs_absZ_Ohm':abs(obs),f'CH{ch}_fit_absZ_Ohm':abs(z)})
                bode=block if bode is None else bode.merge(block,on='Frequency_Hz',how='outer',sort=False)
            name='origin_bode_overlay.csv' if rep==primary else f'origin_bode_overlay_rep{rep}.csv'
            bode.sort_values('Frequency_Hz',ascending=False).to_csv(folder/name,index=False)
        plot_group(folder,curves,key,control=control)
        log(f'Exported channel curves and Origin tables for {key}.')
    export_multidevice_summary(out,groups,control)
    pd.DataFrame(errors).to_csv(out/'channel_metrics.csv',index=False)
    metric_frame=pd.DataFrame(errors)
    if len(metric_frame):
        device_metrics=(metric_frame.groupby('dataset',as_index=False)
            .agg(relative_complex_RMS_percent=('relative_complex_RMS_percent',lambda x:float(np.sqrt(np.mean(np.square(x))))),
                 magnitude_log10_RMS=('magnitude_log10_RMS','mean'),phase_RMSE_deg=('phase_RMSE_deg','mean')))
        device_metrics.to_csv(out/'independent_device_metrics.csv',index=False)
    pd.DataFrame(geometry).to_csv(out/'geometry_mesh_audit.csv',index=False)
    pd.DataFrame(meshcheck).to_csv(out/'mesh_convergence.csv',index=False)
    sensitivity,identifiability=diagnostics(problem,x)
    pd.DataFrame(sensitivity,columns=['parameter','value','lower','upper','lower_distance_decades','upper_distance_decades','nearest_bound','near_bound','log_parameter_sensitivity','note']).to_csv(out/'parameter_sensitivity.csv',index=False)
    write_json(out/'identifiability.json',identifiability)
    write_json(out/'network_profile_verification.json',profile_audit)
    for row in manifest:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('An input file changed during fitting.')
    hits=[r['parameter'] for r in sensitivity if r['near_bound']]
    maxmesh=max((r['max_relative_change_percent'] for r in meshcheck),default=0)
    notes=['Model: geometry-resolved mixed-conduction network',f'Created: {timestamp()}',f'Termination: {info["termination"]}',
        'Objective: sum of per-device mean relative complex squared errors; independent devices have equal weight.',
        'The default fit uses replicate 1 from each of four independent devices; repeated scans are technical quality-control measurements, not independent n.',
        'Material parameters are joint estimates shared across devices; measurement Rs is device-specific. Parameter uncertainty requires device-level resampling or independent-device fits.',
        'Relative complex RMS = 100*sqrt(mean(abs(Zfit-Zobs)^2/abs(Zobs)^2)); phase and band errors also exported.',
        'Boundary hits: '+(', '.join(hits) or 'none'),f'Numerically weak directions: {identifiability["weak_directions"]}',
        f'Maximum 2x mesh change: {maxmesh:.6g}%',
        'No global optimum certificate. Local sensitivity is not a confidence interval.',
        'Continuous mixed-conduction network spans overlap and all bulk sections; the distal ionic node is the exposed-tip terminal.',
        'ICH keeps equipotential Au in overlap and an ionic-only bulk. No PEDOT bulk conversion.',
        'Overlap ion-electronic coupling uses Randles-W area density; dual bulk coupling uses Randles-W volume density.',
        'Tip: (Rt + semi-infinite W) || CPE; all coefficients are area-specific.',
        'Neck walls are encapsulated. Only distal end-face is exposed. No sidewall leakage.',
        'Model settings: '+json.dumps(project['options'],ensure_ascii=False,sort_keys=True),
        'Dual-conducting overlap uses an equipotential Au bus, distributed Au-PEDOT contact, PEDOT rail, and ionic rail.',
        'Znetwork is a coupled network, not additive separately fitted Zmetal+Zbulk for dual materials.',
        'A low residual does not prove transferable or uniquely identified material parameters.',
        'If mesh change exceeds experimental precision, refine the mesh and refit before interpreting parameters.']
    (out/'fit_report.txt').write_text('\n'.join(notes),encoding='utf-8')

    paths=['project.json','parameters.json','input_manifest.json']
    write_json(out/'prediction_lock.json',dict(locked_at=timestamp(),model_id=MODEL_ID,
        model_code_sha256=sha(Path(__file__).with_name('unified.py')),
        files={name:sha(out/name) for name in paths},target_observations_used_for_fitting=False))

def predict_result(result_folder, output_root=None, compare=False, log=print):
    from .project import load_project, write_json, sha, timestamp, read_spectrum
    root=Path(result_folder);project=load_project(root/'project.json')
    lock=json.loads((root/'prediction_lock.json').read_text(encoding='utf-8'))
    if lock['model_id']!=MODEL_ID:raise ValueError('Locked model mismatch.')
    if lock.get('model_code_sha256')!=sha(Path(__file__).with_name('unified.py')):raise ValueError('Prediction rejected because the solver does not match the lock.')
    for name,digest in lock['files'].items():
        if sha(root/name)!=digest:raise ValueError('Prediction rejected because a locked file changed.')
    saved=json.loads((root/'parameters.json').read_text(encoding='utf-8'))
    out=Path(output_root or root)/datetime.now().strftime('prediction_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    f=np.geomspace(.1,1e6,321); predictions=[];parameters=saved['material_values'];measurements=saved['values']
    for row in project['datasets']:
        if row['training']:continue
        if row['material'] not in parameters:raise ValueError(f"No fitted parameters are available for {row['material']}.")

        spec=project.get('prediction_measurements',{}).get(row['batch'],{})
        rs=measurements.get(f'measurement/{row["batch"]}/Rs',spec.get('Rs'))
        if rs is None:raise ValueError(f"No series resistance is defined for prediction batch {row['batch']}.")
        dest=out/row['id'];dest.mkdir()
        for ch,g in enumerate(row['geometry'],1):
            z,_=spectrum(row['material'],parameters[row['material']],g,f,project['options'],rs)
            frame(f,z).to_csv(dest/f'ch{ch}_prediction.txt',sep='\t',index=False)
        predictions.append((row,rs,dest))
    if not predictions:raise ValueError('The project has no held-out datasets.')
    write_json(out/'before_validation_lock.json',dict(locked_at=timestamp(),source_lock=lock,
        predicted_curves={str(p.relative_to(out)):sha(p) for p in out.rglob('*prediction.txt')},optimizer_calls=0))
    metrics_rows=[];manifest=[]
    for row,rs,dest in predictions:
        curves=[]
        for ch,g in enumerate(row['geometry'],1):
            obs=None;ff=f
            if compare:
                if row['replicate']=='all':raise ValueError('Prediction validation requires one explicit replicate.')
                path=Path(row['data_dir'])/f'ch{ch}-{row["replicate"]}.txt'
                digest=sha(path);ff,obs=read_spectrum(path);manifest.append(dict(path=str(path),sha256=digest))
            z,_=spectrum(row['material'],parameters[row['material']],g,ff,project['options'],rs)
            frame(ff,z,obs).to_csv(dest/f'ch{ch}_comparison.txt',sep='\t',index=False)
            curves.append((ch,1,ff,z,obs))
            if obs is not None:metrics_rows.append(dict(dataset=row['id'],channel=ch,**metrics(z,obs)))
        plot_group(dest,curves,row['id'],prediction=True);log(f'Exported locked prediction for {row["id"]}.')
    pd.DataFrame(metrics_rows).to_csv(out/'validation_metrics.csv',index=False)
    write_json(out/'validation_manifest.json',manifest)
    for record in manifest:
        if sha(record['path'])!=record['sha256']:raise RuntimeError('A validation input changed during export.')
    return out

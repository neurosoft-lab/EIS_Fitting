from __future__ import annotations
import copy, os, queue, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from .studio import (DISPLAY, MATERIALS, STUDIES, STUDY_LABELS, PARAM_LABELS,
    app_data_root, group_path, make_project, geometry_layout, read_bundle,
    result_quality, run_study, default_reference, register_reference)
from .project import read_spectrum
from .run_control import RunControl
from .unified import MODEL_ID

def choose_dir(variable, title):
    initial = variable.get() if Path(variable.get()).is_dir() else None
    value = filedialog.askdirectory(title=title, initialdir=initial)
    if value: variable.set(value)

class Preview(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent); self.curves=[]
        top=ttk.Frame(self); top.pack(fill='x',padx=8,pady=6)
        self.title=tk.StringVar(value='Open a result to preview fitted and observed spectra.')
        ttk.Label(top,textvariable=self.title).pack(side='left')
        self.channel=tk.StringVar(value='Overlay')
        picker=ttk.Combobox(top,textvariable=self.channel,state='readonly',width=12,
                            values=['Overlay','Eight panels']+[f'CH{i}' for i in range(1,9)])
        picker.pack(side='right'); picker.bind('<<ComboboxSelected>>',lambda _e:self.draw())
        self.book=ttk.Notebook(self); self.book.pack(fill='both',expand=True)
        self.plots=[]
        for label in ('Bode magnitude','Bode phase','Nyquist'):
            frame=ttk.Frame(self.book); self.book.add(frame,text=label)
            figure=Figure(figsize=(10,6),dpi=100,layout='constrained'); canvas=FigureCanvasTkAgg(figure,master=frame)
            canvas.get_tk_widget().pack(fill='both',expand=True); NavigationToolbar2Tk(canvas,frame).update()
            self.plots.append((figure,canvas))
        self.book.bind('<<NotebookTabChanged>>',lambda _e:self.draw())
    def load(self, result):
        bundle=read_bundle(result); quality=result_quality(result)
        status='pass' if quality['mesh_passed'] else 'review'
        self.title.set(f"{bundle['row']['id']} | RMS {quality['RMS_percent']:.2f}% | mesh {quality['mesh_max_percent']:.3g}% ({status})")
        self.curves=bundle['curves']; self.draw()
    def draw(self):
        if not self.curves:return
        kind=self.book.index('current'); figure,canvas=self.plots[kind]; figure.clear()
        split=self.channel.get()=='Eight panels'; axes=list(figure.subplots(2,4).flat) if split else [figure.subplots()]
        curves=self.curves if self.channel.get() in ('Overlay','Eight panels') else [c for c in self.curves if c['channel']==int(self.channel.get()[2:])]
        colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#7B3294','#777777']
        for curve in curves:
            ax=axes[curve['channel']-1] if split else axes[0]; f=curve['frequency']; obs=curve['obs']; fit=curve.get('fit'); color=colors[curve['channel']-1]
            if kind==0:
                ax.loglog(f,abs(obs),'o',ms=3,mfc='none',color=color,label=f"CH{curve['channel']} observed")
                if fit is not None:ax.loglog(f,abs(fit),color=color,label=f"CH{curve['channel']} fitted")
            elif kind==1:
                ax.semilogx(f,np.angle(obs,deg=True),'o',ms=3,mfc='none',color=color,label=f"CH{curve['channel']} observed")
                if fit is not None:ax.semilogx(f,np.angle(fit,deg=True),color=color,label=f"CH{curve['channel']} fitted")
            else:
                ax.plot(obs.real,-obs.imag,'o',ms=3,mfc='none',color=color,label=f"CH{curve['channel']} observed")
                if fit is not None:ax.plot(fit.real,-fit.imag,color=color,label=f"CH{curve['channel']} fitted")
                ax.set_aspect('equal',adjustable='datalim')
            if split:ax.set_title(f"CH{curve['channel']}")
        for ax in axes:
            ax.grid(alpha=.2); ax.set_xlabel('Frequency (Hz)' if kind<2 else "Z' (ohm)")
            ax.set_ylabel(['|Z| (ohm)','Phase (deg)',"-Z'' (ohm)"][kind])
        if not split:axes[0].legend(ncol=4,fontsize=8)
        canvas.draw_idle()

class App(tk.Tk):
    def __init__(self, app):
        super().__init__(); self.app=Path(app).resolve(); self.title('EIS Fitting')
        self.geometry(f'{min(1380,self.winfo_screenwidth()-60)}x{min(900,self.winfo_screenheight()-90)}'); self.minsize(960,640)
        self.events=queue.Queue(); self.control=RunControl(); self.busy=False; self.last_result=None
        self.study=tk.StringVar(value='Overlap'); self.material=tk.StringVar(value=DISPLAY['DCH'])
        self.data=tk.StringVar(); self.output=tk.StringVar(); self.layout=None; self.project=None
        self.search={name:tk.StringVar(value=str(value)) for name,value in {'starts':24,'batches':2,'max_nfev':2000,'workers':4}.items()}
        style=ttk.Style(self); style.theme_use('clam'); style.configure('Treeview',rowheight=25)
        self.book=ttk.Notebook(self); self.book.pack(fill='both',expand=True,padx=8,pady=8)
        self._build_fit(); self._build_batch(); self._build_results(); self._build_log()
        footer=ttk.Frame(self); footer.pack(fill='x',padx=8,pady=6)
        self.status=tk.StringVar(value='Ready | CPU float64')
        ttk.Label(footer,textvariable=self.status).pack(side='left')
        ttk.Button(footer,text='Stop',command=self.stop).pack(side='right')
        self.pause_button=ttk.Button(footer,text='Pause',command=self.toggle_pause,state='disabled'); self.pause_button.pack(side='right',padx=8)
        self.protocol('WM_DELETE_WINDOW',self.close); self.after(150,self.poll); self.refresh()

    def _build_fit(self):
        tab=ttk.Frame(self.book); self.book.add(tab,text='Single fit')
        top=ttk.Frame(tab); top.pack(fill='x',padx=10,pady=8)
        ttk.Label(top,text='Study').grid(row=0,column=0,sticky='w',padx=4)
        study=ttk.Combobox(top,textvariable=self.study,state='readonly',values=list(STUDIES),width=15); study.grid(row=0,column=1,padx=4)
        ttk.Label(top,text='Material').grid(row=0,column=2,sticky='w',padx=4)
        material=ttk.Combobox(top,textvariable=self.material,state='readonly',values=list(DISPLAY.values()),width=20); material.grid(row=0,column=3,padx=4)
        study.bind('<<ComboboxSelected>>',self.change_model); material.bind('<<ComboboxSelected>>',self.change_model)
        ttk.Label(top,text='Data').grid(row=1,column=0,sticky='w',padx=4,pady=6)
        ttk.Entry(top,textvariable=self.data,width=85).grid(row=1,column=1,columnspan=3,sticky='ew',padx=4)
        ttk.Button(top,text='Browse',command=lambda:choose_dir(self.data,'Select EIS data folder')).grid(row=1,column=4,padx=4)
        ttk.Label(top,text='Output').grid(row=2,column=0,sticky='w',padx=4)
        ttk.Entry(top,textvariable=self.output,state='readonly',width=85).grid(row=2,column=1,columnspan=4,sticky='ew',padx=4)
        top.columnconfigure(3,weight=1)
        search=ttk.Frame(tab); search.pack(fill='x',padx=10,pady=3)
        for col,(key,var) in enumerate(self.search.items()):
            ttk.Label(search,text=key.replace('_',' ').title()).grid(row=0,column=2*col,padx=(3,1))
            ttk.Entry(search,textvariable=var,width=8).grid(row=0,column=2*col+1,padx=(1,8))
        self.params=ttk.Treeview(tab,columns=('value','lower','upper','unit'),show='tree headings',height=14)
        self.params.heading('#0',text='Parameter'); self.params.column('#0',width=310)
        for col,title,width in [('value','Initial',120),('lower','Lower',120),('upper','Upper',120),('unit','Unit',170)]:
            self.params.heading(col,text=title); self.params.column(col,width=width)
        self.params.pack(fill='both',expand=True,padx=10,pady=6); self.params.bind('<Double-1>',self.edit_parameter)
        buttons=ttk.Frame(tab); buttons.pack(fill='x',padx=10,pady=8)
        ttk.Button(buttons,text='Refresh',command=self.refresh).pack(side='left')
        ttk.Button(buttons,text='Preview data',command=self.preview_data).pack(side='left',padx=8)
        ttk.Button(buttons,text='Run',command=self.run_single).pack(side='right')

    def _build_batch(self):
        tab=ttk.Frame(self.book); self.book.add(tab,text='Batch fit')
        self.batch_tree=ttk.Treeview(tab,columns=('run','material','study','path'),show='headings',height=16)
        for col,title,width in [('run','Run',60),('material','Material',150),('study','Study',100),('path','Data',720)]:
            self.batch_tree.heading(col,text=title); self.batch_tree.column(col,width=width)
        for study in STUDIES:
            for material in MATERIALS:
                iid=study+'_'+material.replace(' ','_'); path=group_path(app_data_root(self.app),study,material)
                self.batch_tree.insert('','end',iid=iid,values=('Yes',DISPLAY[material],study,str(path)))
        self.batch_tree.pack(fill='both',expand=True,padx=10,pady=10); self.batch_tree.bind('<Double-1>',self.toggle_batch)
        ttk.Button(tab,text='Run selected',command=self.run_batch).pack(anchor='e',padx=12,pady=8)

    def _build_results(self):
        self.preview=Preview(self.book); self.book.add(self.preview,text='Results')
        tools=ttk.Frame(self.preview); tools.pack(fill='x',before=self.preview.book)
        ttk.Button(tools,text='Open result',command=self.open_result).pack(side='left',padx=6)
        ttk.Button(tools,text='Open folder',command=lambda:self.open_folder(self.last_result)).pack(side='left',padx=6)

    def _build_log(self):
        tab=ttk.Frame(self.book); self.book.add(tab,text='Log')
        self.logtext=tk.Text(tab,wrap='word',font=('Consolas',10)); self.logtext.pack(fill='both',expand=True)

    def material_key(self): return next(key for key,value in DISPLAY.items() if value==self.material.get())
    def change_model(self,_event=None): self.data.set('');self.refresh()
    def refresh(self):
        root=self.data.get() or str(group_path(app_data_root(self.app),self.study.get(),self.material_key()))
        self.project=make_project(self.app,self.study.get(),self.material_key(),root,require_data=False)
        self.data.set(root); self.output.set(self.project['output_root']); self.layout=geometry_layout(self.project)
        self.params.delete(*self.params.get_children())
        for name,(value,lower,upper,unit) in self.layout.entries.items():
            bare=name.split('/')[-1]; label=PARAM_LABELS.get(bare,bare)
            self.params.insert('','end',iid=name,text=f'{name} | {label}',values=(f'{value:.8g}',f'{lower:.8g}',f'{upper:.8g}',unit))
    def edit_parameter(self,_event):
        selected=self.params.selection()
        if not selected:return
        key=selected[0]; values=self.params.item(key,'values'); win=tk.Toplevel(self); win.title(key)
        variables=[tk.StringVar(value=values[i]) for i in range(3)]
        for row,(label,var) in enumerate(zip(('Initial','Lower','Upper'),variables)):
            ttk.Label(win,text=label).grid(row=row,column=0,padx=8,pady=6); ttk.Entry(win,textvariable=var).grid(row=row,column=1,padx=8)
        def save():
            try:
                value,lower,upper=map(lambda v:float(v.get()),variables)
                if not 0<lower<=value<=upper or lower>=upper:raise ValueError('Require 0 < lower <= initial <= upper.')
                self.project.setdefault('parameters',{})[key]={'value':value,'lower':lower,'upper':upper}
                self.params.item(key,values=(f'{value:.8g}',f'{lower:.8g}',f'{upper:.8g}',values[3])); win.destroy()
            except Exception as exc:messagebox.showerror('Invalid parameter',str(exc),parent=win)
        ttk.Button(win,text='Save',command=save).grid(row=3,column=0,columnspan=2,pady=8)
    def apply_search(self,project):
        for key,var in self.search.items(): project['search'][key]=int(var.get())
        return project
    def preview_data(self):
        try:
            curves=[]
            project=make_project(self.app,self.study.get(),self.material_key(),self.data.get())
            for row in project['datasets']:
                for ch in range(1,9):
                    path=Path(row['data_dir'])/f'ch{ch}-1.txt'; f,z=read_spectrum(path)
                    curves.append(dict(dataset=row['id'],device=row.get('device',row['id']),channel=ch,frequency=f,obs=z,fit=None))
            self.preview.curves=curves; self.preview.title.set(f'{self.study.get()} | {self.material.get()} | observed data'); self.preview.draw(); self.book.select(self.preview)
        except Exception as exc:messagebox.showerror('Data preview',str(exc))
    def run_single(self):
        try:
            project=make_project(self.app,self.study.get(),self.material_key(),self.data.get())
            project['parameters']=copy.deepcopy(self.project.get('parameters',{}))
            self.apply_search(project)
            reference='' if self.study.get()=='Overlap' else default_reference(self.app,self.material_key())
            def work(log,control):
                result=run_study(project,'auto',reference,3.,log,control)
                if project['datasets'][0]['study']=='Overlap' and not control.is_set():register_reference(self.app,result)
                return result
            self.launch(work)
        except Exception as exc:messagebox.showerror('Cannot start',str(exc))
    def toggle_batch(self,_event):
        selected=self.batch_tree.selection()
        if selected:
            values=list(self.batch_tree.item(selected[0],'values')); values[0]='No' if values[0]=='Yes' else 'Yes'; self.batch_tree.item(selected[0],values=values)
    def run_batch(self):
        tasks=[]
        for iid in self.batch_tree.get_children():
            enabled,display,study,path=self.batch_tree.item(iid,'values')
            if enabled=='Yes': tasks.append((study,next(k for k,v in DISPLAY.items() if v==display),path))
        if not tasks:messagebox.showinfo('Batch fit','Select at least one task.');return
        try: search_values={key:int(var.get()) for key,var in self.search.items()}
        except Exception as exc:messagebox.showerror('Cannot start',str(exc));return
        def work(log,control):
            references={}; last=None
            for study,material,path in tasks:
                if control.is_set():break
                project=make_project(self.app,study,material,path);project['search'].update(search_values)
                reference='' if study=='Overlap' else references.get(material) or default_reference(self.app,material)
                log(f'Starting {study} | {DISPLAY[material]}')
                last=run_study(project,'auto',reference,3.,log,control)
                if study=='Overlap' and not control.is_set():register_reference(self.app,last); references[material]=str(last)
                self.events.put(('preview',last))
            return last
        self.launch(work)
    def launch(self,call):
        if self.busy:messagebox.showinfo('Busy','Wait for the current run or stop it.');return
        self.busy=True; self.control.clear(); self.pause_button.configure(state='normal',text='Pause'); self.status.set('Running'); self.book.select(self.logtext.master)
        def worker():
            try:self.events.put(('done',call(lambda text:self.events.put(('log',text)),self.control)))
            except Exception:self.events.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='log':self.logtext.insert('end',str(value)+'\n');self.logtext.see('end')
                elif kind=='preview':self.show_result(value)
                elif kind=='done':
                    self.busy=False;self.control.resume();self.pause_button.configure(state='disabled');self.status.set('Completed')
                    if value:self.show_result(value)
                elif kind=='error':
                    self.busy=False;self.control.set();self.pause_button.configure(state='disabled');self.status.set('Failed')
                    self.logtext.insert('end',value+'\n');messagebox.showerror('Run failed',value[-1800:])
        except queue.Empty:pass
        self.after(150,self.poll)
    def stop(self):
        if self.busy:self.control.set();self.status.set('Stopping after the current safe checkpoint')
    def toggle_pause(self):
        if not self.busy:return
        if self.control.is_paused():self.control.resume();self.pause_button.configure(text='Pause');self.status.set('Running')
        else:self.control.pause();self.pause_button.configure(text='Resume');self.status.set('Pausing at a safe checkpoint')
    def show_result(self,path):
        try:self.preview.load(path);self.last_result=Path(path);self.book.select(self.preview)
        except Exception as exc:messagebox.showerror('Cannot open result',str(exc))
    def open_result(self):
        path=filedialog.askdirectory(title='Select a completed result package')
        if path:self.show_result(path)
    def open_folder(self,path):
        if path and Path(path).exists():os.startfile(str(Path(path).resolve()))
    def close(self):
        if self.busy:
            if messagebox.askyesno('Run active','Stop the run before closing?'):self.stop()
            return
        self.destroy()

SingleApp=App
def main(app,kind='single'):
    App(app).mainloop()

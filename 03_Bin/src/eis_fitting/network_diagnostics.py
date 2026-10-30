import numpy as np
from scipy.linalg import solve_banded
from .unified import spectrum,prepare_mesh,ionic_edge_impedance

def network_profile(material,p,g,frequency,options):
    f=np.asarray(frequency,float);mesh=prepare_mesh(g,options.get('mesh_step_um',100.))
    _,components=spectrum(material,p,g,f,options)
    jw=2j*np.pi*f;n=len(mesh.axial)+1
    yo=1/(p['rct_overlap']+p['warburg_overlap']/np.sqrt(jw))+jw*p['cdl_overlap']
    ri,_=ionic_edge_impedance(p,mesh,f,options)
    def edge(values,k,j=None):
        if np.ndim(values)==1:return values[k]
        return values[:,k] if j is None else values[j,k]
    if material=='ICH':
        ve=np.full((len(f),n),np.nan+0j);vi=np.empty((len(f),n),complex)
        ve[:,:mesh.overlap_edges+1]=(components['Znetwork']+components['Ztip'])[:,None]
        vi[:,-1]=components['Ztip']
        for k in range(n-2,mesh.overlap_edges-1,-1):vi[:,k]=vi[:,k+1]+edge(ri,k)
        yr=yo[:,None]*mesh.overlap_area
        transfer=np.zeros((len(f),n),complex)


        count=mesh.overlap_edges
        for j in range(len(f)):
            gg=1/(ri[:count] if np.ndim(ri)==1 else ri[j,:count])
            diagonal=yr[j,:count].copy();diagonal+=gg
            diagonal[1:]+=gg[:-1]
            ab=np.zeros((3,count),complex);ab[1]=diagonal
            if count>1:ab[0,1:]=-gg[:-1];ab[2,:-1]=-gg[:-1]
            rhs=np.zeros(count,complex);rhs[-1]=gg[-1]*components['Zoverlap'][j]
            difference=np.r_[solve_banded((1,1),ab,rhs),components['Zoverlap'][j]]
            vi[j,:count+1]=ve[j,:count+1]-difference
            transfer[j,:count+1]=yr[j,:count+1]*difference
        error=np.max(abs(transfer.sum(axis=1)-1))
        ie=np.full((len(f),n-1),np.nan+0j)
    else:
        re=p['rho_e']*mesh.axial.copy()
        yb=1/(p['rct_bulk']+p['warburg_bulk']/np.sqrt(jw))+jw*p['cdl_bulk']
        yr=yo[:,None]*mesh.overlap_area+yb[:,None]*mesh.bulk_volume
        ve=np.empty((len(f),n),complex);vi=np.zeros((len(f),n),complex)
        contact=mesh.overlap_area/p['rc_au_pedot_area']
        reconstructed=[]
        for j in range(len(f)):


            dim=2*n-1;Y=np.zeros((dim,dim),complex);rhs=np.zeros(dim,complex)
            def connect(a,b,gg):
                Y[a,a]+=gg
                if b is not None:Y[b,b]+=gg;Y[a,b]-=gg;Y[b,a]-=gg
            for k in range(n):
                e=2*k;i=2*k+1 if k<n-1 else None;connect(e,i,yr[j,k])
                if contact[k]:Y[e,e]+=contact[k];rhs[e]+=contact[k]
            for k in range(n-1):
                connect(2*k,2*k+2,1/re[k]);connect(2*k+1,2*k+3 if k+1<n-1 else None,1/edge(ri,k,j))
            v=np.linalg.solve(Y,rhs);znet=1/np.sum(contact*(1-v[::2]));reconstructed.append(znet)
            ve[j]=v[::2]*znet+components['Ztip'][j]
            vi[j,:-1]=v[1::2]*znet+components['Ztip'][j];vi[j,-1]=components['Ztip'][j]
        error=np.max(abs(np.asarray(reconstructed)-components['Znetwork'])/abs(components['Znetwork']))
        transfer=yr*(ve-vi);ie=(ve[:,:-1]-ve[:,1:])/re
    ii=(vi[:,:-1]-vi[:,1:])/(ri if np.ndim(ri)==2 else ri[None,:])
    if material!='ICH':

        pass
    if not np.isfinite(error) or error>1e-4:
        raise FloatingPointError(f'Independent nodal verification failed ({error:g}).')
    rows=[]
    for j,frequency in enumerate(f):
        for k,x in enumerate(mesh.x_um):
            rows.append(dict(Frequency_Hz=frequency,x_um=x,
                Ve_real_V_per_A=float(ve[j,k].real),Ve_imag_V_per_A=float(ve[j,k].imag),
                Vi_real_V_per_A=float(vi[j,k].real),Vi_imag_V_per_A=float(vi[j,k].imag),
                transverse_current_real_A_per_A=float(transfer[j,k].real),transverse_current_imag_A_per_A=float(transfer[j,k].imag),
                electronic_axial_real_A_per_A=float(ie[j,k].real) if k<n-1 else float('nan'),
                ionic_axial_real_A_per_A=float(ii[j,k].real) if k<n-1 else float('nan')))
    return rows,float(error)

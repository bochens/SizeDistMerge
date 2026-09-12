"""Small, QC-screened optical-change experiment; never writes production files.

Reuse only the settings and align/combine function from the R2 notebook.
Replay historical native bins with the historical optical code and old LUTs,
and verify them against saved NetCDF arrays. Fit each new case with the same
previous-minute R2 temporal prior (not a new sequential campaign history).
QC uses the existing paired-QC population, source checks, cost < 0.2, and the
frozen previous R2 CPC warning bounds, with current inlet-filtered CPC values.
These four deliberately selected cases are not a campaign impact estimate.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sizedistmerge-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/sizedistmerge-numba')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from arcsix_production import arcsix_merge_production as mp
from sizedistmerge.utils import select_between, remap_dndlog_by_edges, remap_dndlog_by_edges_any
from sizedistmerge.ict_utils import read_inlet_flag, drop_timezone_from_index

NAMES = ('FIMS', 'UHSAS', 'POPS', 'APS')
PARAMS = ('retrieved_uhsas_n_fit', 'retrieved_pops_n_fit', 'retrieved_aps_density')
COLORS = dict(FIMS='#d7485b', UHSAS='#198c59', POPS='#e88b16', APS='#306fbd', Merged='black')
TARGETS = ('2024-05-28 11:59:19', '2024-06-05 11:37:35',
           '2024-06-13 11:37:48', '2024-07-30 17:30:16')


def notebook_scope(path):
    cells = {c.get('id'): ''.join(c.get('source', []))
             for c in json.loads(path.read_text())['cells']}
    # Suppress the settings cell's campaign-path announcement: no campaign runs.
    scope = dict(Path=Path, np=np, REPO=REPO, mp=mp, print=lambda *a, **k: None)
    # Do not execute inventory, runner, orchestration, export, or plotting cells.
    for key in ('r2-settings', 'r2-align-combine'):
        exec(compile(cells[key], f'{path}:{key}', 'exec'), scope)
    return scope


def index_for(ds, start, end):
    base = pd.Timestamp(ds.attrs['base_time_iso'])
    a = base + pd.to_timedelta(ds.time_start_since_base_s.values, unit='s')
    b = base + pd.to_timedelta(ds.time_end_since_base_s.values, unit='s')
    idx = np.flatnonzero((a == pd.Timestamp(start)) & (b == pd.Timestamp(end)))
    assert len(idx) == 1, (start, end, idx)
    return int(idx[0])


def load_legacy(path):
    spec = importlib.util.spec_from_file_location('optical_pre_fix', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare(args, out):
    out.mkdir(parents=True, exist_ok=True)
    source = REPO / 'notebooks/arcsix_1min_merged_sizedist_production_r2.ipynb'
    (out / 'r2_notebook_snapshot.ipynb').write_bytes(source.read_bytes())
    legacy = subprocess.check_output(['git', 'show', '5f77fa45:src/sizedistmerge/optical_diameter.py'], cwd=REPO)
    (out / 'historical_optical_diameter.py').write_bytes(legacy)
    (out / 'production_module_snapshot.py').write_bytes((REPO/'arcsix_production/arcsix_merge_production.py').read_bytes())
    (out / 'combine_module_snapshot.py').write_bytes((REPO/'src/sizedistmerge/combine.py').read_bytes())
    population = pd.read_csv(REPO/'outputs/qc_fims_consensus_survey/qc_clear_fims_minutes.csv', dtype={'day': str})
    metadata = []
    for start in TARGETS:
        selected = population[population.start == start]
        assert len(selected) == 1, f'Case not in paired QC-clear FIMS population: {start}'
        row = selected.iloc[0]
        day, date, end = row.day, start[:10], row.end
        frames = mp.read_arcsix_merge_instruments_for_day(date, args.data_root/'LARGE-APS', args.data_root/'FIMS',
            uhsas_dir=args.data_root/'PUTLS-UHSAS', pops_dir=args.data_root/'PUTLS-POPS', require_fims=False)
        # Match production's independent FIMS-presence filter before its lag
        # correction. require_fims=True would instead intersect every instrument
        # at unshifted timestamps and change the averaged air samples. Actual
        # FIMS use is enforced below and by reference_source_flag, not this switch.
        frames = {k: drop_timezone_from_index(v) for k, v in frames.items()}
        frames['FIMS'].index -= pd.Timedelta(seconds=10)
        flags = read_inlet_flag(args.data_root/'LARGE-InletFlag', start=date, prefix='ARCSIX')
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        chunks = mp.filter_chunk_by_inlet_flag({k: v.loc[a:b] for k,v in frames.items()}, flags, a, b, 10)
        assert chunks['FIMS'].QC_Flag.eq(0).all() and len(chunks['FIMS']) >= 10
        assert all(len(chunks[k]) >= 10 for k in NAMES)
        specs = mp.make_filtered_specs(chunks, out/'source_averaging.log')[0]
        assert not mp.source_zero_run_qc(specs, reference_source_flag=0)
        saved = {}
        for name in NAMES:
            saved['Before_'+name+'_edges'], saved['Before_'+name+'_y'] = specs[name][1:3]
        paths = {v: root/day/date/f'{date}_sizedist_merged.nc' for v,root in [('R1',args.r1_root),('R2',args.r2_root)]}
        for version, path in paths.items():
            with xr.open_dataset(path) as ds:
                i = index_for(ds, start, end)
                assert int(ds.reference_source_flag.values[i]) == 0
                assert float(ds.optimization_best_cost.values[i]) < .2
                saved[version+'_theta'] = np.array([ds[k].values[i] for k in PARAMS])
                saved[version+'_cost'] = ds.optimization_best_cost.values[i]
                saved[version+'_common_edges'] = ds.fine_edges_nm.values
                for name in (*NAMES, 'Merged'):
                    saved[version+'_'+name+'_common_y'] = ds[name.lower()+'_dNdlogDp' if name=='Merged' else name.lower()+'_aligned_dNdlogDp'].values[i]
                if version == 'R2':
                    assert ds.combination_use_consensus == 0 and ds.combination_tikhonov_lambda == 2e-5
                    assert ds.pops_source_ri_real == 1.615 and ds.pops_source_ri_imag == .001
                    assert ds.response_bins_fit == ds.response_bins_apply == 100
                    assert ds.qc_wide_source_zero_run.values[i] == 0
                    # Read the exact prior saved by the optimizer, not a guessed row.
                    period = int(ds.period_idx.values[i])
                    history = np.load(path.parent/'optimizer_info'/f'period_{period:04d}.npz')
                    saved['prior'] = history['temporal_target'].copy()
                    np.testing.assert_allclose(history['temporal_weights'], [.1,.1,5e-7])
        cpc = mp._read_cpc_series(args.data_root/'LARGE-MICROPHYSICAL', date, 'CNgt10nm')
        cpc = mp._filter_cpc_for_qc(cpc, args.data_root, date, 10)
        saved['cpc'] = float(cpc.loc[a:b].median())
        assert np.isfinite(saved['cpc'])
        qc_path = Path(str(args.r1_root)+'_R2_adjacent_zeros')/'qc_flagged_nc'/f'{date}_sizedist_merged.nc'
        with xr.open_dataset(qc_path) as ds:
            comment = ds.warning_merged_gt10_diff_from_cpc.attrs['comment']
            saved['cpc_low'] = float(re.search(r'r_low_warn=([^,]+)', comment)[1])
            saved['cpc_high'] = float(re.search(r'r_high_warn=([^\.]+\.[0-9]+)', comment)[1])
        key = f'{day}_{int(row.period):04d}'
        np.savez_compressed(out/f'{key}_inputs.npz', **saved)
        metadata.append(dict(key=key, day=day, start=start, end=end, period=int(row.period),
                             source_seconds={k:len(chunks[k]) for k in NAMES}, paths={k:str(v) for k,v in paths.items()}))
        print('Prepared QC-clear FIMS case', start, flush=True)
    manifest = dict(cases=metadata, previous_R2_root=str(args.r2_root), old_lut_dir=str(args.old_lut_dir),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        temporal_prior='Each case uses the exact saved previous-R2 optimizer prior; independent one-minute experiment.',
        QC='Paired old-product QC clear, source FIMS QC=0, no wide zero runs, cost<0.2, frozen earlier R2 CPC warning bounds.',
        not_campaign_validation=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return metadata


def combine(scope, native):
    args = dict(lam=scope['LAMBDA_TIK'], n_points=scope['FINE_BIN'], alpha_fims=1,
        alpha_uhsas=1, alpha_pops=.5, alpha_aps=1.5, use_consensus=False,
        data_space='log10', weight_profiles=scope['COMBINATION_WEIGHT_PROFILES'],
        ignore_nonpositive_source=True, preserve_zero_endpoints={'FIMS':'first','APS':'last'})
    for name, tag in zip(NAMES, ('fims_sel','uhsas_fit','pops_fit','aps_fit')):
        args['e_'+tag], args['y_'+tag] = native[name]
    spec = next(iter(mp.make_consensus_merged_spec(**args)[0].values()))
    return spec[1], spec[2]


def one_case(job):
    meta, output, legacy_dir = job
    out = Path(output); key = meta['key']
    scope = notebook_scope(out/'r2_notebook_snapshot.ipynb')
    saved = dict(np.load(out/f'{key}_inputs.npz'))
    specs = {n: (np.sqrt(saved['Before_'+n+'_edges'][:-1]*saved['Before_'+n+'_edges'][1:]),
                 saved['Before_'+n+'_edges'], saved['Before_'+n+'_y'], np.full_like(saved['Before_'+n+'_y'],np.nan)) for n in NAMES}
    _, f_edges, f_y, _ = select_between(*specs['FIMS'], xmin=10, xmax=500)
    old = load_legacy(out/'historical_optical_diameter.py')
    old_luts = {n:old.SigmaLUT(str(Path(legacy_dir)/file)) for n,file in
                [('UHSAS','uhsas_sigma_col_1054nm.zarr'),('POPS','pops_sigma_col_405nm.zarr')]}
    for v in ('R1','R2'):
        native={'FIMS':(f_edges,f_y)}
        for n,p in zip(('UHSAS','POPS','APS'), saved[v+'_theta']):
            e,y = saved['Before_'+n+'_edges'], saved['Before_'+n+'_y']
            if n=='APS': ef=mp._aps_remap_fn(e,[p])
            else:
                src=complex(1.52) if n=='UHSAS' or v=='R1' else complex(1.615,.001)
                ef=old.convert_do_lut(e,src,complex(p),old_luts[n],response_bins=120 if v=='R1' else 100)
            native[n]=(ef,remap_dndlog_by_edges(e,ef,y))
        for n,(e,y) in native.items():
            check=remap_dndlog_by_edges_any(e,saved[v+'_common_edges'],y)
            np.testing.assert_allclose(check,saved[v+'_'+n+'_common_y'],rtol=1e-8,atol=1e-10,equal_nan=True,
                                       err_msg=f'{key} {v} {n} historical native replay')
            saved[v+'_'+n+'_edges'],saved[v+'_'+n+'_y']=e,y
        if v=='R2':
            e,y=combine(scope,native)
            np.testing.assert_allclose(remap_dndlog_by_edges_any(e,saved[v+'_common_edges'],y),
                saved[v+'_Merged_common_y'],rtol=1e-7,atol=1e-10,equal_nan=True)
    print(key,'historical native bins and latest R2 combination verified',flush=True)
    begin=time.monotonic()
    result=scope['align_and_combine'](specs,{}, {}, saved['prior'].tolist(),complex(1.615,.001),(100,100),reference_source_flag=0)
    fit=result['fit']; saved['New_theta']=result['next_prior']; saved['New_cost']=fit['best_cost']
    common=saved['R2_common_edges']; saved['New_common_edges']=common
    for n in (*NAMES,'Merged'):
        e,y=result['aligned']['Reference' if n=='FIMS' else n]
        saved['New_'+n+'_edges'], saved['New_'+n+'_y']=e,y
        saved['New_'+n+'_common_y']=remap_dndlog_by_edges_any(e,common,y)
        if n in ('UHSAS','POPS','APS'):
            np.testing.assert_allclose(y*np.diff(np.log10(e)),
                saved['Before_'+n+'_y']*np.diff(np.log10(saved['Before_'+n+'_edges'])),rtol=2e-14,atol=1e-12,equal_nan=True)
    records=[]
    # Restrict quantitative version comparisons to identical observed support.
    for v in ('R1','R2'):
        np.testing.assert_allclose(saved[v+'_common_edges'],common,rtol=0,atol=1e-10)
    shared=np.logical_and.reduce([np.isfinite(saved[v+'_Merged_common_y']) for v in ('R1','R2','New')])
    for v in ('R1','R2','New'):
        e,y=saved[v+'_common_edges'],saved[v+'_Merged_common_y']
        total=float(mp.integrate_dndlog_gt_cutoff(y[None,:],e)[0])
        residual=total-float(saved['cpc'])
        qc=bool(float(saved[v+'_cost'])<.2 and saved['cpc_low']<=residual<=saved['cpc_high'])
        saved[v+'_qc_clear']=qc
        rec=dict(key=key,start=meta['start'],version=v,uhsas_n=float(saved[v+'_theta'][0]),
                 pops_n=float(saved[v+'_theta'][1]),density=float(saved[v+'_theta'][2]),
                 cost=float(saved[v+'_cost']),total_number=total,
                 shared_total_number=float(np.sum(y[shared]*np.diff(np.log10(e))[shared])),
                 shared_bins=int(shared.sum()),cpc=float(saved['cpc']),cpc_residual=residual,qc_clear=qc)
        for tag,lo,hi in [('N20_100',20,100),('N100_300',100,300),('N300_1000',300,1000),('N1000_3000',1000,3000)]:
            widths=np.maximum(0,np.log10(np.minimum(e[1:],hi)/np.maximum(e[:-1],lo)))
            rec[tag]=float(np.sum(y[shared]*widths[shared]))
        records.append(rec)
    saved['history_total']=np.asarray(fit['hist']['total'])
    np.savez_compressed(out/f'{key}_results.npz',**saved)
    diag=dict(fit['optimizer_diagnostics'],elapsed_seconds=time.monotonic()-begin,
              data_cost=fit['data_cost'],temporal_cost=fit['temporal_cost'])
    (out/f'{key}_optimizer.json').write_text(json.dumps(diag,indent=2)+'\n')
    print(key,'NEW LUT fit complete',diag,flush=True)
    return records


def plots(out,metadata):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    kept=[]
    for meta in metadata:
        data=dict(np.load(out/f'{meta["key"]}_results.npz'))
        if all(bool(data[v+'_qc_clear']) for v in ('R1','R2','New')):kept.append((meta,data))
    assert kept,'No fully QC-clear comparison cases'
    for part in range(0,len(kept),2):
        group=kept[part:part+2]
        fig,axes=plt.subplots(len(group),4,figsize=(15,3.6*len(group)),squeeze=False)
        fig.subplots_adjust(left=.055,right=.995,bottom=.14,top=.90,hspace=.66,wspace=.17)
        for row,(meta,data) in enumerate(group):
            for col,(v,title) in enumerate(zip(('Before','R1','R2','New'),('Before conversion','R1','Previous R2','Corrected optics'))):
                ax=axes[row,col]
                ax.set_title(title)
                for n in NAMES:
                    e,y=data[v+'_'+n+'_edges'],data[v+'_'+n+'_y']
                    ax.stairs(np.where(y>0,y,np.nan),e,color=COLORS[n],lw=1.1,baseline=None)
                if v!='Before':
                    e,y=data[v+'_common_edges'],data[v+'_Merged_common_y']
                    ax.stairs(np.where(y>0,y,np.nan),e,color='black',lw=1.8,baseline=None)
                    p=data[v+'_theta']
                    ax.text(0,-.27,f'n(UHSAS)={p[0]:.3f}; n(POPS)={p[1]:.3f}\nρ(APS)={p[2]:.0f} kg m⁻³',transform=ax.transAxes,fontsize=8,color='.35',va='top')
                ax.set(xscale='log',yscale='log',xlim=(10,5000),ylim=(1e-3,5e3),xlabel='Diameter (nm)')
                ax.grid(alpha=.18,which='major')
                if col==0:
                    ax.set_ylabel(r'$dN/d\log_{10}D_p$ (cm$^{-3}$)')
                    ax.text(0,1.19,meta['start']+' UTC',transform=ax.transAxes,fontsize=9)
                else: ax.tick_params(labelleft=False)
        fig.legend([Line2D([],[],color=COLORS[n],lw=1.8) for n in (*NAMES,'Merged')],(*NAMES,'Merged'),
                   loc='upper center',bbox_to_anchor=(.5,1),ncol=5,frameon=False)
        for ext in ('png','pdf'):fig.savefig(out/f'comparison_{part//2+1}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)


def summarize_saved(out, metadata):
    """Recompute common-support number changes without rerunning any fit."""
    rows=[]; differences=[]
    for meta in metadata:
        d=dict(np.load(out/f'{meta["key"]}_results.npz'))
        e=d['R2_common_edges']
        for v in ('R1','New'):np.testing.assert_allclose(d[v+'_common_edges'],e,rtol=0,atol=1e-10)
        shared=np.logical_and.reduce([np.isfinite(d[v+'_Merged_common_y']) for v in ('R1','R2','New')])
        case_rows={}
        for v in ('R1','R2','New'):
            y=d[v+'_Merged_common_y'];p=d[v+'_theta']
            row=dict(key=meta['key'],start=meta['start'],version=v,
                uhsas_n=float(p[0]),pops_n=float(p[1]),density=float(p[2]),cost=float(d[v+'_cost']),
                qc_clear=bool(d[v+'_qc_clear']),shared_bins=int(shared.sum()),
                shared_total_number=float(np.sum(y[shared]*np.diff(np.log10(e))[shared])))
            for tag,lo,hi in [('N20_100',20,100),('N100_300',100,300),('N300_1000',300,1000),('N1000_3000',1000,3000)]:
                dw=np.maximum(0,np.log10(np.minimum(e[1:],hi)/np.maximum(e[:-1],lo)))
                row[tag]=float(np.sum(y[shared]*dw[shared]))
            rows.append(row);case_rows[v]=row
        for v in ('R1','R2'):
            row=dict(start=meta['start'],baseline=v)
            for name in ('shared_total_number','N20_100','N100_300','N300_1000','N1000_3000'):
                a,b=case_rows[v][name],case_rows['New'][name]
                row[name+'_delta']=b-a;row[name+'_pct']=100*(b/a-1) if a>0 else np.nan
            for name in ('uhsas_n','pops_n','density'):row[name+'_delta']=case_rows['New'][name]-case_rows[v][name]
            differences.append(row)
    pd.DataFrame(rows).to_csv(out/'number_and_retrieval_comparison.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'paired_changes.csv',index=False)
    (out/'analysis_script.sha256').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r1-root',type=Path,required=True);p.add_argument('--r2-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--old-lut-dir',type=Path,default=REPO/'tmp/legacy_luts_before_20260912')
    p.add_argument('--output',type=Path,default=REPO/'outputs/corrected_optics_merge_tests')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true')
    p.add_argument('--plots-only',action='store_true')
    args=p.parse_args();out=args.output
    if args.resume or args.plots_only:metadata=json.loads((out/'manifest.json').read_text())['cases']
    else:metadata=prepare(args,out)
    jobs=[(m,str(out),str(args.old_lut_dir)) for m in metadata]
    records=[]
    if not args.plots_only:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for block in pool.map(one_case,jobs):records.extend(block)
    summarize_saved(out,metadata)
    plots(out,metadata)


if __name__=='__main__':main()

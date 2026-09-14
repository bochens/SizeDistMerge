import numpy as np
import pytest
from sizedistmerge.combine import native_bin_consensus_multipliers, merge_sizedists_bin_totals
from sizedistmerge.uncertainty import bin_total_log_sensitivity


def series():
    e=np.geomspace(100,1000,9)
    return [dict(name=n,edges=e,number=h*np.diff(np.log10(e)),alpha=1.)
            for n,h in [('a',np.full(8,10.)),('b',np.full(8,12.)),('c',np.full(8,1.))]]


def test_suppression_and_weakening():
    s=series();a=native_bin_consensus_multipliers(s,2);b=native_bin_consensus_multipliers(s,8)
    assert np.all(a[2]<a[0])
    for x,y in zip(a,b):assert np.all(x<=y) and np.all(x>0) and np.all(y<=1)
    for w in native_bin_consensus_multipliers(s[:2],2):np.testing.assert_allclose(w,1)
    s[2]['number'][:]=0
    for w in native_bin_consensus_multipliers(s,2):np.testing.assert_allclose(w,1)


def test_exact_agreement_and_invalid_strength():
    s=series();s[1]['number']=s[0]['number'].copy()
    w=native_bin_consensus_multipliers(s,2)
    np.testing.assert_allclose(w[0],1);np.testing.assert_allclose(w[1],1)
    np.testing.assert_allclose(w[2],1e-12)
    for c in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError):native_bin_consensus_multipliers(s,c)


def test_uncertainty_derivative_uses_effective_fixed_weights():
    s=series();edges=np.geomspace(100,1000,15)
    mult=native_bin_consensus_multipliers(s,2)
    s=[dict(v,alpha=m) for v,m in zip(s,mult)]
    y,d=merge_sizedists_bin_totals(edges,s,lam=3e-6)
    K,keys,_=bin_total_log_sensitivity(s,d,edges)
    direction=np.sin(np.arange(len(keys)))
    curves=[]
    for sign in [-1,1]:
        changed=[dict(v,number=v['number'].copy()) for v in s]
        lookup={v['name']:v for v in changed}
        for k,(name,i) in enumerate(keys):lookup[name]['number'][i]*=10**(sign*.001*direction[k])
        # Multipliers deliberately held fixed, as in the reported conditional uncertainty.
        fit,_=merge_sizedists_bin_totals(edges,changed,lam=3e-6)
        curves.append(np.log10(fit))
    np.testing.assert_allclose((curves[1]-curves[0])/.002,K@direction,atol=.025)

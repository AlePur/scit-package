from ._summary import summarize_multiple_modalities, membership_summary, _sortable_data

def function(adata):
    scit.tl.layer_pick_features(cttag, lc_meth, marker_group)
    basemask = lc_meth.cached_mask.copy()
    basefeatures = lc_meth.get_feature_names()
    summ = membership_summary(cttag, lc_meth, 'label', use_cached_mask=True)
    # data = _sortable_data({'default': summ}, sorted_l=ordered_l, return_structured=True)
    # assert data.shape[1] == 1
    # maxs = np.argmax(data[:,0,:],axis=1)

    data = {}
    #for i in range(len(ordered_l)):
    for c in me_class['clusters'].unique().to_numpy():
        print("=" * 30)
        print(c)
        print("-" * 10)
        flist = me_class.filter(pl.col('clusters') == c)['name'].to_numpy()
        print(len(flist))
        if len(flist) < 10:
            continue


        d = summarize_multiple_modalities([cttag, cttag, rna], [lc_meth, lc_acet, lc_rna], flist, 'label',
                                        norm_per_modality=False)
        # d_emb = summarize_multiple_modalities([rna], [lc_rna], _fmask, 'label', sqrt_transform=False)

        scit.set_defaults(figsize=(4,3))

        keys = list(d.keys())

        # t = lambda x: np.log1p(10*x)
        
        for i,k in enumerate(keys):
            #  d[k][l].mean(axis=0).A1 gives mean over cells in cell type l
            means = np.array([(d[k][l].mean(axis=0).A1) for l in ordered_l])

            # now means is array of (cell type, gene mean)
            # median gives (cell type, median over gene means)
            data[f'{c}_{k}'] = np.median(means, axis=1)
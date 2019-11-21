model_versions = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model_names = {
    'distilgpt2': 'distill-GPT2',
    'gpt2': 'GPT2-small',
    'gpt2-medium': 'GPT2-medium',
    'gpt2-large': 'GPT2-large',
    'gpt2-xl': 'GPT2-XL'
}
filters = ['filtered', 'unfiltered']
source_to_variations = {
    'winobias': ['dev', 'test'],
    'winogender': ['bls', 'bergsma'],
}
for model_version in model_versions:
    for source in ['winobias', 'winogender']:
        variations = source_to_variations[source]
        for variation in variations:
            fnames = []; fig_names = []
            for filter in filters:
                fnames.append(f"images/heat_maps_with_bar_indirect/{source}_{model_version}_{filter}_{variation}.pdf")
                fig_names.append(f"indirect_heatmap_{source}_{model_version}_{filter}_{variation}")
            if source == 'winobias':
                split = variation
                caption = f"Mean indirect effect for {model_names[model_version]} on winobias dataset, {split} set. Left: filtered, right: unfiltered."
            else:
                stat = variation
                caption = f"Mean indirect effect for {model_names[model_version]} on winogender dataset, using {stat.upper()} as stereotypicality measure. Left: filtered, right: unfiltered."
            print(f"""
\\begin{{figure*}}
\\centering
\\begin{{minipage}}[b]{{.4\\textwidth}}
\\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 1.5cm 0.5cm}},clip]{{{fnames[0]}}}
\\label{{{fig_names[0]}}}
\\end{{minipage}}\\qquad
\\begin{{minipage}}[b]{{.4\\textwidth}}
\\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 1.5cm 0.5cm}},clip]{{{fnames[1]}}}
\\label{{{fig_names[1]}}}
\\end{{minipage}}
\\vspace{{-1em}}
\\caption{{{caption}}}
\\vspace{{-1em}}
\\end{{figure*}}
""")

            # print(f"""
            # \\begin{{figure*}}
            # \\centering
            # \\begin{{minipage}}[b]{{.4\\textwidth}}
            # \\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 1.5cm 0.5cm}},clip]{{{fnames[0]}}}
            # \\caption{{{captions[0]}}}\\label{{{fig_names[0]}}}
            # \\end{{minipage}}\\qquad
            # \\begin{{minipage}}[b]{{.4\\textwidth}}
            # \\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 1.5cm 0.5cm}},clip]{{{fnames[1]}}}
            # \\caption{{{captions[1]}}}\\label{{{fig_names[1]}}}
            # \\end{{minipage}}
            # \\end{{figure*}}
            # """)
                # latex = f"""
                # \\begin{{figure}}[t]
                #     \\centering
                #     \\includegraphics[width=\\linewidth]{{{fname}}}
                #     \\caption{{{caption}}}
                #     \\label{{fig:{fig_name}}}
                # \\end{{figure}}
                # """
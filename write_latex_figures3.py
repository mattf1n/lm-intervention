model_versions = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model_names = {
    'distilgpt2': 'distill-GPT2',
    'gpt2': 'GPT2-small',
    'gpt2-medium': 'GPT2-medium',
    'gpt2-large': 'GPT2-large',
    'gpt2-xl': 'GPT2-XL'
}
variation_names = {
    'bls': 'BLS',
    'bergsma': 'Bergsma',
    'dev': 'Dev',
    'test': 'Test'
}

filters = ['filtered', 'unfiltered']
source_to_variations = {
    'winobias': ['dev', 'test'],
    'winogender': ['bls', 'bergsma'],
}
model_version = 'gpt2'
for source in ['winobias', 'winogender']:
    variations = source_to_variations[source]
    fnames = []; fig_names = []
    caption = f"Indirect effect for {source.capitalize()} ({model_names[model_version]}). Left: filt., right: "\
              f"unfilt. Top: {variation_names[variations[0]]}, bottom: {variation_names[variations[1]]}."
    for variation in variations:
        for filter in filters:
            fnames.append(f"images/heat_maps_with_bar_indirect/{source}_{model_version}_{filter}_{variation}.pdf")
            fig_names.append(f"indirect_heatmap_{source}_{model_version}_{filter}_{variation}")
    print(f"""
\\begin{{figure*}}
\\centering
\\begin{{minipage}}[b]{{.4\\textwidth}}
\\vspace{{-1.3em}}
\\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 0.5cm 0.7cm}},clip]{{{fnames[0]}}}
\\label{{{fig_names[0]}}}
\\end{{minipage}}
\\begin{{minipage}}[b]{{.4\\textwidth}}
\\vspace{{-1.3em}}
\\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 0.5cm 0.7cm}},clip]{{{fnames[1]}}}
\\label{{{fig_names[1]}}}
\\end{{minipage}}
\\begin{{minipage}}[b]{{.4\\textwidth}}
\\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 0.5cm 0.7cm}},clip]{{{fnames[2]}}}
\\label{{{fig_names[2]}}}
\\end{{minipage}}
\\begin{{minipage}}[b]{{.4\\textwidth}}
\\includegraphics[width=1\\linewidth,trim={{0.5cm 0.5cm 0.5cm 0.7cm}},clip]{{{fnames[3]}}}
\\label{{{fig_names[3]}}}
\\end{{minipage}}
\\vspace{{-1.6em}}
\\caption{{{caption}}}
\\vspace{{-2em}}
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
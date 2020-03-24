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
source = 'winobias'
variation='dev'
filter = 'filtered'

latex = ""
for model_version in model_versions:
    caption = f"Mean indirect effect for {model_names[model_version]} on {source.capitalize()} for heads(the heatmap) and layers (the column chart)."
    fname = f"images/heat_maps_with_bar_indirect/{source}_{model_version}_{filter}_{variation}.pdf"
    fig_name = f"indirect_heatmap_{source}_{model_version}_{filter}_{variation}"
    latex += f"""\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=1\\linewidth]{{{fname}}}
    \\caption{{{caption}}}
    \\label{{fig:{fig_name}}}
\\end{{figure}}"""

print(latex)

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
#import matplotlib

light_blue='#87cefa'
dark_cyan='#008b8b'
lime='#9acd32'
dark_blue='#056098'
grey='#575757'
black='#000000'
color_palette=[light_blue,dark_cyan,lime,dark_blue, grey]


color_palette_4=color_palette[:4]
stat_color_mapping={s:c for s,c in zip (['mean','min','max','std'], color_palette_4)}


#my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','#008b8b','#9acd32','#e3f8b7'])

data_dir='new_data'

affine_max_instance_id=5
affine_alphas=[0.25,0.5,0.75]
bbob_max_instance_id=100
maximum_generations_to_run_algorithms=50
m4_problem_count=5000
random_problem_count=5000
all_benchmarks=['bbob','affine','random','m4']
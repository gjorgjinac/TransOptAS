import matplotlib
color_palette=['#87cefa','#008b8b','#9acd32','#056098','#575757']


color_palette_4=['#87cefa','#008b8b','#9acd32','#056098','#575757']
stat_color_mapping={s:c for s,c in zip (['mean','min','max','std'], color_palette_4)}


my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','#008b8b','#9acd32','#e3f8b7'])


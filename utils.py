import matplotlib as mpl

def set_mplstyle():
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['font.sans-serif'] = ['STIXGeneral']
    mpl.rcParams['xtick.direction'] = "in"
    mpl.rcParams['ytick.direction'] = "in"
    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["legend.framealpha"] = 1
    mpl.rcParams["font.size"] = 26
    mpl.rcParams["figure.autolayout"] = True
    mpl.rcParams["errorbar.capsize"] = 5

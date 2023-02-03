# Script for creating lime info record for packaging.

import pickle
import sys
from numpy import prod, dtype
from datetime import datetime

def main(NT, NS, NC=3, ND=4, gauge_t=False, out_file='info.p'):
    
    NT = int(NT)
    NS = int(NS)
    NC = int(NC)
    ND = int(ND)
    
    DTYPE = '>c16'
    FORTRAN_ORDER = False
    
    if gauge_t:
        SHAPE = (NT, NS, NS, NS, NC, NC)
    elif not gauge_t:
        SHAPE = (NT, NS, NS, NS, ND, NC, NC)
    else:
        raise Exception("Invalid gauge_t input.")
    
    info_dict = {'dtype': DTYPE,
                 'shape': SHAPE,
                 'nbytes': prod(SHAPE)*dtype(DTYPE).itemsize,
                 'fortran_order': FORTRAN_ORDER,
                 'misc': datetime.now().strftime("generated at %H:%m:%S on %D")}
    
    with open(out_file, 'wb') as handle:
        pickle.dump(info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return 0
    
    
if __name__ == "__main__":
    main(*sys.argv[1:])
    

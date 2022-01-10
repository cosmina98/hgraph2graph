from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
from iter_gen import IterGenDirect, StockParamLoader
from hgraph import MolGraph

def in_vocab(smiles, vocab, avocab):
    mg = MolGraph(smiles)
    for idx, pair in mg.mol_tree.nodes(data='label'):
        # pair consists of two types SMILES of the same mol: (unmapped, root_mapped)
        if pair not in vocab.vmap:
            return False
    for idx, element_symbol in mg.mol_graph.nodes(data='label'):
        if element_symbol not in avocab.vmap:
            return False

    return True

if __name__ == '__main__':
    import argparse
    import os
    from pathlib import Path
    parser = argparse.ArgumentParser(prog='sdf encoder')
    parser.add_argument("sdf",metavar = 'F', type=Path, 
                        help="path to input file (sdf)")
    parser.add_argument("-o","--output", type=Path,
                        help="directory for output, \
                        default to ./${input_filename_base}.npy")
    p = parser.parse_args()
    
    if not p.sdf.exists() or p.sdf.is_dir():
        raise FileNotFoundError("[Invalid Input File] {}".format(p.sdf))

    if p.output is None:
        # if nothing supplied, keep orignal file path and stem
        npy_file = p.sdf.with_suffix('.npy')
    elif p.output.is_dir():
        # if only a directory is supplied and it exists, attache the file stem
        npy_file = p.output.joinpath(p.sdf.with_suffix('.npy'))
    else:
        # everything else is considered as a file path NOT a dir path
        npy_file = p.output.with_suffix('.npy')
    
    mols = [m for m in Chem.SDMolSupplier(str(p.sdf)) if m]
    smiles = [Chem.MolToSmiles(m) for m in mols]
    print('Number of Molecules loaded: {}'.format(len(mols)))

    model_loc = "./ckpt/chembl-pretrained/model.ckpt" 
    vocab_loc = './data/chembl/vocab.txt'
    batch_size = 500
    steps = 5
    # n_samples = 10
    torch.manual_seed(42)
    eta = 1
    prms = StockParamLoader(model_loc, vocab_loc)
    
    batch_generator = IterGenDirect(batch_size, steps, prms, eta)

    size = len(smiles)
    step = 100
    idx = list(range(0,size,step))
    idx.append(size)
    batches = list(zip(idx[0:-1],idx[1:]))

    z_vecs = []
    all_cleaned = []

    for start, stop in batches:
        print(start, stop)
        cleaned = list(filter(lambda smi: in_vocab(smi, batch_generator.vocab, batch_generator.atom_vocab), smiles[start:stop]))
        all_cleaned.extend(cleaned)

        with torch.no_grad():
            cur_z_vecs = batch_generator.encode_smiles(cleaned)
        z_vecs.append(cur_z_vecs.to('cpu'))
        torch.cuda.empty_cache()
    final_z_vecs = torch.cat(z_vecs)
    print("Vectors of shape {} generated".format(final_z_vecs.shape))

    np.save(npy_file, final_z_vecs.numpy())

    with npy_file.with_suffix('.smi').open('w') as f:
        for smi in all_cleaned:
            f.write(smi+'\n')

    print("Vectors saved at {}".format(npy_file))

    

    
        
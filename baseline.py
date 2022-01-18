from iter_gen import IterGen, StockParamLoader
from rdkit import Chem
import torch
from MARSPlus.evaluator.scorer import Scorer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class CombinedScorer(Scorer):
    def __init__(self):
        config = {
            'objectives' : ['jnk3','qed','sa']
        }
        config['score_wght'], config['score_succ'], config['score_clip'] = {}, {}, {}
        for obj in config['objectives']:
            if   obj == 'gsk3b' or obj == 'jnk3': wght, succ, clip = 1.0, 0.5, 0.6
            elif obj == 'qed'                   : wght, succ, clip = 1.0, 0.6, 0.7
            elif obj == 'sa'                    : wght, succ, clip = 1.0, .67, 0.7
            config['score_wght'][obj] = wght
            config['score_succ'][obj] = succ
            config['score_clip'][obj] = clip
        super().__init__(config)
    def get_weighted_scores(self, mol):
        weighted_scores, _ = self.get_scores([mol])
        return weighted_scores[0]

    def get_weighted_scores_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        weighted_scores, _ = self.get_scores([mol])
        return weighted_scores[0]

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(prog='Baseline VAE Generator')
    parser.add_argument("-o","--output", type=Path,
                        help="output file path, \
                        default to ./vae_gen_(rate).csv")
    parser.add_argument("-s","--steps", type=int, default = 2000,
                        help="number of steps to perform, \
                        default to 2000")
    parser.add_argument("-b","--batch_size", type=int, default = 500,
                        help="number of molecules to generate in one step, \
                        default to 500") 
    parser.add_argument("-n","--n_samples", type=int, default = 10,
                        help="number of samples per seeding molecules, \
                        default to 10")
    parser.add_argument("-r","--rate", type=float, default = 0.01,
                        help="acceptance rate for low scoring offsprings, \
                        default to 0.01")
    parser.add_argument("-m","--mode", type=str,
                        help="mode of generation, available options: \
                        ('rand','direct','convert')")
    parser.add_argument("-e","--eta", type=int, default = 1,
                        help="scaling factor for sampling width, \
                        default to 1")  
    parser.add_argument("-d","--seed", type=int,
                        help="seed for random number generators")

    p = parser.parse_args()

    if p.output is None:
        # if nothing supplied, set to default value
        out_file = "./vae_gen_{}.csv".format(p.rate)
    elif p.output.is_dir():
        # if only a directory is supplied and it exists, attache the default file name
        out_file = p.output.joinpath("vae_gen_{}.csv".format(p.rate))
    else:
        out_file = p.output
    
    combined_scorer = CombinedScorer()

    oracle = combined_scorer.get_weighted_scores_from_smiles
    model_loc = "./ckpt/chembl-pretrained/model.ckpt" 
    vocab_loc = './data/chembl/vocab.txt'
    prms = StockParamLoader(model_loc, vocab_loc)
    if p.seed is not None:
        torch.manual_seed(p.seed)
    with torch.no_grad():
        
        generator = IterGen(
            batch_size = p.batch_size, 
            steps = p.steps,
            n_samples = p.n_samples,
            prms = prms,
            eta = p.eta,
            oracle = oracle,
            gen_mode = p.mode,
            acc_rate = p.rate,
            save_path = out_file
        )

        for batch in generator:
            print("Step", generator.step)
            info = list(zip(*batch))
            for i, info in enumerate(info):
                print(i, '\t', round(info[0],3), '\t', info[1])
            print('***'*10)

from hgraph import HierVAE, common_atom_vocab, PairVocab, MolGraph
from hgraph.hgnn import make_cuda
import torch

class StockParamLoader:
    def __init__(self, model_loc, vocab_loc):
        self.model_loc = model_loc
        vocab = [x.strip("\r\n ").split() for x in open(vocab_loc)]
        self.vocab = PairVocab(vocab)
        ###### values below are specific to the stock ChEMBL model #######
        self.atom_vocab = common_atom_vocab
        self.rnn_type = "LSTM"
        self.hidden_size = 250
        self.embed_size = 250
        self.latent_size = 32
        self.depthT = 15
        self.depthG = 15
        self.diterT = 1
        self.diterG = 3
        self.dropout = 0.0

class _IterGenBase(HierVAE):
    def __init__(self, batch_size, steps, model_prms, eta, seed = None):
        super().__init__(model_prms)
        self.cuda()
        self.load_state_dict(torch.load(model_prms.model_loc)[0])
        self.eval()
        if seed:
            torch.manual_seed(seed)
        self.vocab = model_prms.vocab
        self.atom_vocab = model_prms.atom_vocab
        self.batch_size = batch_size
        self.step = 0 
        self.max_step = steps
        # multiplication factor of latent vector perturbation
        self.eta = eta
        # Initial batch is randomly sampled
        self.rand_sample()

    def rand_sample(self):
        self.z_vecs = torch.randn(self.batch_size, self.latent_size).cuda()

    def decode_smiles(self, z_vecs):
        z_vecs_3 = (z_vecs, z_vecs, z_vecs)
        return self.decoder.decode(z_vecs_3, greedy=True, max_decode_step=150)

    def generate_new_vecs(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if self.step < self.max_step:
            self.generate_new_vecs()
            self.cur_batch = self.decode_smiles(self.z_vecs)
            self.step += 1
            return self.cur_batch
        else:
            raise StopIteration

class IterGenRand(_IterGenBase):
    def __init__(self, batch_size, steps, prms, eta, seed = None):
        super().__init__(batch_size, steps, prms, eta, seed)

    def generate_new_vecs(self):
        """
        Randomly sample the latent space
        """
        self.rand_sample()

class IterGenDirect(_IterGenBase):
    def __init__(self, batch_size, steps, prms, eta, seed = None):
        super().__init__(batch_size, steps, prms, eta, seed)

    def generate_new_vecs(self):
        """
        Generation is anchored on current z_vecs
        Directly perturb the current latent vectors without encoding from SMILES
        """
        z_log_var = -torch.abs( torch.log(self.z_vecs.var(0)) )
        epsilon = torch.randn_like(self.z_vecs).cuda()
        self.z_vecs = self.z_vecs + torch.exp(z_log_var / 2) * epsilon * self.eta

class IterGenConvert(_IterGenBase):
    def __init__(self, batch_size, steps, prms, eta, seed = None):
        super().__init__(batch_size, steps, prms, eta, seed)

    def generate_new_vecs(self):
        """
        Generation is anchored on current SMILES batch
        Encode the current batch of SMILES and perturb the latent vectors
        """
        _, tensorized, _ = MolGraph.tensorize(self.cur_batch, self.vocab, self.atom_vocab)
        tree_tensors, graph_tensors = make_cuda(tensorized)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        self.z_vecs, _ = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=True)

if __name__=='__main__':
    # Stock ChEMBL model
    model_loc = "./ckpt/chembl-pretrained/model.ckpt" 
    vocab_loc = './data/chembl/vocab.txt'
    batch_size = 10
    steps = 20
    seed = 42
    eta = 1
    prms = StockParamLoader(model_loc, vocab_loc)

    batch_generator = IterGenRand(batch_size, steps, prms, eta, seed)
    for batch in batch_generator:
        print("Step", batch_generator.step)
        print("Total SMILES:", len(batch))
        for i, smi in enumerate(batch):
            print(i, smi)
        

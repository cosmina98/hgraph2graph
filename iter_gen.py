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
    def __init__(self, batch_size, steps, model_prms):
        super().__init__(model_prms)
        self.cuda()
        self.load_state_dict(torch.load(model_prms.model_loc)[0])
        self.eval()
        
        self.vocab = model_prms.vocab
        self.atom_vocab = model_prms.atom_vocab
        self.batch_size = batch_size
        self.step = 0
        self.max_step = steps
        
    def rand_sample(self):
        return torch.randn(self.batch_size, self.latent_size).cuda()

    def perturb_latent(self, z_vecs, eta):
        z_log_var = -torch.abs( torch.log(z_vecs.var(0)) )
        epsilon = torch.randn_like(z_vecs).cuda()
        return z_vecs + torch.exp(z_log_var / 2) * epsilon * eta

    def encode_smiles(self, smiles_list, perturb=False):
        # No purterbation on latent vector for the default behavior
        failed_smiles = []
        tree_batch = []
        for s in smiles_list:
            try:
                tree_batch.append(MolGraph(s))
            except AssertionError:
                failed_smiles.append(s)
        _, tensorized, _ = MolGraph.tensorize(smiles_list, self.vocab, self.atom_vocab)
        tree_tensors, graph_tensors = make_cuda(tensorized)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        z_vecs, _ = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=perturb)
        return z_vecs

    def decode_smiles(self, z_vecs):
        z_vecs_3 = (z_vecs, z_vecs, z_vecs)
        return self.decoder.decode(z_vecs_3, greedy=True, max_decode_step=150)

    def generate_new_vecs(self):
        raise NotImplementedError

    # def filter_select(self, smiles_list, measure_filter):
    #     return filter(measure_filter, smiles_list)

    def __iter__(self):
        # Initial batch is randomly sampled
        self.z_vecs = self.rand_sample()
        self.cur_batch = self.decode_smiles(self.z_vecs)
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
    def __init__(self, batch_size, steps, prms, eta):
        super().__init__(batch_size, steps, prms)

    def generate_new_vecs(self):
        """
        Randomly sample the latent space
        """
        self.z_vecs = self.rand_sample()

class IterGenDirect(_IterGenBase):
    def __init__(self, batch_size, steps, prms, eta):
        super().__init__(batch_size, steps, prms)
        # multiplication factor of latent vector perturbation
        self.eta = eta

    def generate_new_vecs(self):
        """
        Generation is anchored on current z_vecs
        Directly perturb the current latent vectors without encoding from SMILES
        """
        self.z_vecs = self.perturb_latent(self.z_vecs, self.eta)

class IterGenConvert(_IterGenBase):
    def __init__(self, batch_size, steps, prms, eta):
        super().__init__(batch_size, steps, prms)

    def generate_new_vecs(self):
        """
        Generation is anchored on current SMILES batch
        Encode the current batch of SMILES and perturb the latent vectors
        """
        self.z_vecs=self.encode_smiles(self.cur_batch, perturb=True)

# class BatchEncoder():
#     def __init__(self, smiles_list, batch_size, model_prms):
#         super().__init__(model_prms)
#         self.cuda()
#         self.load_state_dict(torch.load(model_prms.model_loc)[0])
#         self.eval()
        
#         self.vocab = model_prms.vocab
#         self.atom_vocab = model_prms.atom_vocab

#         size = len(smiles_list)
#         idx = list(range(0,size,batch_size))
#         idx.append(size)
#         self.batches = list(zip(idx[0:-1],idx[1:]))


#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.step < self.max_step:
#             self.generate_new_vecs()
#             self.cur_batch = self.decode_smiles(self.z_vecs)
#             self.step += 1
#             return self.cur_batch
#         else:
#             raise StopIteration


if __name__=='__main__':
    # Stock ChEMBL model
    model_loc = "./ckpt/chembl-pretrained/model.ckpt" 
    vocab_loc = './data/chembl/vocab.txt'
    batch_size = 10
    steps = 5
    # n_samples = 10
    torch.manual_seed(42)
    eta = 1
    prms = StockParamLoader(model_loc, vocab_loc)
    
    with torch.no_grad():
        batch_generator = IterGenRand(batch_size, steps, prms, eta)
        for batch in batch_generator:
            print("Step", batch_generator.step)
            print("Total SMILES:", len(batch))
            for i, smi in enumerate(batch):
                print(i, smi)
            print('***'*10)
        torch.cuda.empty_cache()

        batch_generator = IterGenDirect(batch_size, steps, prms, eta)
        for batch in batch_generator:
            print("Step", batch_generator.step)
            print("Total SMILES:", len(batch))
            for i, smi in enumerate(batch):
                print(i, smi)
            print('***'*10)
        torch.cuda.empty_cache()

        batch_generator = IterGenConvert(batch_size, steps, prms, eta)
        for batch in batch_generator:
            print("Step", batch_generator.step)
            print("Total SMILES:", len(batch))
            for i, smi in enumerate(batch):
                print(i, smi)
            print('***'*10)
        torch.cuda.empty_cache()
    
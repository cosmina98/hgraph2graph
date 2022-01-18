import numpy as np
from hgraph import HierVAE, common_atom_vocab, PairVocab, MolGraph
from hgraph.hgnn import make_cuda
import torch
import time

conditional_add = lambda x,y : x + y if type(x) is int \
    else (x[0] + y, x[1] + y)

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


class SmilesBatchTensorizor:
    """
    Process list of SMILES into MolGraph objects with vocab checking and
    self cleaning functions
    """
    def __init__(self, smiles, vocab, avocab):
        self.mg_batch = [MolGraph(s) for s in smiles]
        self.vocab = vocab
        self.avocab = avocab

    def vocab_clean(self):
        """
        Clean the MolGraph objects for non-existent SMILES fragment 
        in the vocab. This is to avoid key error during encoding
        The list of failed SMILES are returned
        """
        cleaned = []
        failed = []
        for mg in self.mg_batch:
            if SmilesBatchTensorizor.in_vocab(mg, self.vocab, self.avocab):
                cleaned.append(mg)
            else:
                failed.append(mg.smiles)
        self.mg_batch = cleaned
        return failed

    @staticmethod
    def in_vocab(mol_graph, vocab, avocab):
        """Check if MolGraph nodes SMILES are in vocab"""
        for idx, pair in mol_graph.mol_tree.nodes(data='label'):
            # pair consists of two types SMILES of the same mol: 
            # (unmapped, root_mapped)
            if pair not in vocab.vmap:
                return False
        for idx, element_symbol in mol_graph.mol_graph.nodes(data='label'):
            if element_symbol not in avocab.vmap:
                return False
        return True

    def make_tensors(self):
        """
        Tensorize (one-hot) the MolGraph objects in a batch. 
        Same as the MolGraph.tensorize(), except that this only 
        deal with the instance variable, MolGraph, directly.
        """
        tree_tensors, tree_batchG = MolGraph.tensorize_graph(
            [x.mol_tree for x in self.mg_batch], self.vocab)
        graph_tensors, graph_batchG = MolGraph.tensorize_graph(
            [x.mol_graph for x in self.mg_batch], self.avocab)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = \
            max( [len(c) for x in self.mg_batch for c in x.clusters] )
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['inter_label'] = \
                [(x + offset, y) for x,y in attr['inter_label']]
            tree_batchG.nodes[v]['cluster'] = cls = \
                [x + offset for x in attr['cluster']]
            tree_batchG.nodes[v]['assm_cands'] = \
                [conditional_add(x, offset) for x in attr['assm_cands']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)

        all_orders = []
        for i,hmol in enumerate(self.mg_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) \
                for x,y,z in hmol.order[:-1]] + \
                [(hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
        return (tree_batchG, graph_batchG), \
               (tree_tensors, graph_tensors), \
               all_orders


class VAEInterface(HierVAE):
    def __init__(self, model_prms):
        super().__init__(model_prms)
        self.cuda()
        self.load_state_dict(torch.load(model_prms.model_loc)[0])
        self.eval()
        
        self.vocab = model_prms.vocab
        self.atom_vocab = model_prms.atom_vocab

    def encode_tensorized(self, tree_tensors, graph_tensors, perturb=False):
        """
        Encode the tensorized (one-hot, hidden size) tree and mol graphs
        into latent vectors.
        No purterbation on latent vector by default
        """
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(
            tree_tensors, graph_tensors
            )
        z_vecs, _ = self.rsample(
            root_vecs, self.R_mean, self.R_var, perturb=perturb
            )
        return z_vecs

    def encode_smiles(self, smiles_list, perturb=False, 
                      clean_smiles = True, report_failed = False):
        """
        Encode a batch of SMILES into latent vectors
        No purterbation on latent vector by default
        """
        smiles_batch = SmilesBatchTensorizor(
            smiles_list, self.vocab, self.atom_vocab
            )
        if clean_smiles:
            # Filter out SMILES not in vocab
            failed = smiles_batch.vocab_clean() 
            if report_failed:
                print('{} SMILES failed'.format(len(failed)))
                for f in failed: print(f)
        _, tensorized, _ = smiles_batch.make_tensors()
        tree_tensors, graph_tensors = make_cuda(tensorized)
        root_vecs, tree_vecs, _, graph_vecs = \
            self.encoder(tree_tensors, graph_tensors)
        z_vecs, _ = self.rsample(
            root_vecs, self.R_mean, self.R_var, perturb=perturb
            )
        return z_vecs

    def decode_smiles(self, z_vecs):
        z_vecs_3 = (z_vecs, z_vecs, z_vecs)
        # return a list of SMILES
        # FIXME: decode could fail due to index error 
        # in inc_graph.py line 130
        # seed = 42 and max_decode = 150 to reproduce this error
        try:
            return self.decoder.decode(
                z_vecs_3, greedy=True, max_decode_step=100
            )
        except IndexError:
            print('Failed to decode')
            print('Use Random generated vecs instead')
            new_z_vecs = torch.randn(z_vecs.shape[0], self.latent_size).cuda()
            return self.decode_smiles(new_z_vecs)

class VAEGenerator(VAEInterface):
    def __init__(self, model_prms, 
                 oracle, sort_descending = True):
        super().__init__(model_prms)

        self.oracle = oracle
        self.sort_descending = sort_descending

    def perturb_latent(self, z_vecs, eta):
        z_log_var = -torch.abs( torch.log(z_vecs.var(0)) )
        epsilon = torch.randn_like(z_vecs).cuda()
        return z_vecs + torch.exp(z_log_var / 2) * epsilon * eta

    def sample_vecs_from_anchors(self, z_vecs, n_samples, eta = 1):
        """
        Randomly sample latent vectors around 
        the given latent vectors (anchors).
        Perturbation of vectors using -abs(log(variance)) as constraint.
        returned vecs shape (batch_size, n_samples, latent_size)
        """
        expanded = z_vecs.reshape(z_vecs.shape[0],1,-1).repeat(1,n_samples,1)
        expanded = self.perturb_latent(expanded, eta)
        return expanded

    def decode_expanded(self, expanded_vecs):
        """
        Decode the latent vector with shape: 
        (batch_size, n_samples, latent_size)

        return a list of lists of SMILES with shape: 
        (batch_size, n_samples)
        """
        expanded_smiles = []
        for v in expanded_vecs:
            expanded_smiles.append(self.decode_smiles(v))
        return expanded_smiles

    def pick_best_sample(self, smiles_list, oracle, descending = True):
        """
        Pick the best SMILES according to the oracle scores, 
        return the index and the best SMILES
        """
        
        scores = [(oracle(smiles), i, smiles) \
            for i, smiles in enumerate(smiles_list) if smiles]
        # scores: (oracle_score, index, smiles)
        scores.sort(reverse = descending)
        # for x in scores:
        #     print(x)
        return scores[0]

    def generate_new_vecs(self):
        raise NotImplementedError

    def generate_new_smiles(self, 
                            z_vecs, n_samples, oracle, 
                            descending = True):
        candidate_vecs = self.sample_vecs_from_anchors(z_vecs, n_samples)
        candidate_smiles = self.decode_expanded(candidate_vecs)

        # for smi_list in candidate_smiles:
        #     for i, smi in enumerate(smi_list):
        #         print(i, smi)

        selected_scores, ids, selected_smiles = \
            zip(*(self.pick_best_sample(smi_list, oracle, descending) \
            for smi_list in candidate_smiles))
        
        ids = torch.LongTensor(ids).cuda()
        # reshape indices into (batch_size, 1, latent_size) 
        # for gather the best latent vectors from candidates from dim 1
        ids = ids.unsqueeze(-1).repeat(1,candidate_vecs.shape[-1]).unsqueeze(1)
        # select the best ones and sqeeze vectors back into 
        # (batch_size, latent_size)
        selected_vecs = candidate_vecs.gather(1,ids).squeeze(1)

        return np.array(selected_scores), selected_vecs, selected_smiles


class IterGen(VAEGenerator):
    def __init__(self, batch_size, steps, n_samples ,prms, eta, oracle,
                 gen_mode, acc_rate = 0.01, save_path = None):
        super().__init__(prms,oracle,sort_descending=True) 
        self.n_samples = n_samples
        self.eta = eta
        self.acc_rate = acc_rate
        self.all_smiles = []
        self.all_scores = []
        self.batch_size = batch_size
        self.step = 0
        self.max_step = steps
        self.save_path = save_path
        if self.save_path is not None:
            with open(self.save_path,'w') as f:
                f.write('STEP,SCORE,SMILES\n')
        # Initial batch is randomly sampled
        self.z_vecs = self.rand_sample_vecs()
        self.cur_batch = self.decode_smiles(self.z_vecs)
        self.cur_scores = np.zeros(batch_size)
        self.assign_gen_mode(gen_mode)

    def rand_sample_vecs(self):
        return torch.randn(self.batch_size, self.latent_size).cuda()

    def convert_sample_vecs(self):
        """
        Generation is anchored on current SMILES batch
        Encode the current batch of SMILES 
        and perturb the latent vectors
        """
        return self.encode_smiles(self.cur_batch, perturb=True)

    def direct_sample_vecs(self):
        """
        Generation is anchored on current z_vecs.
        Directly perturb the current latent vectors 
        without encoding from SMILES.
        """
        return self.perturb_latent(self.z_vecs, self.eta)

    def assign_gen_mode(self, gen_mode):
        allowed_mode = {
            'rand' : self.rand_sample_vecs,
            'direct' : self.direct_sample_vecs,
            'convert' : self.convert_sample_vecs
            }
        if gen_mode not in allowed_mode.keys():
            print('Invalid Generation Mode! Selection has to be from',
                  allowed_mode.keys())
        else:
            self.generate_new_vecs = allowed_mode[gen_mode]

    def accept_new(self, new_scores, new_smiles, rate = 0.05):
        accepted = np.logical_or(
            new_scores > self.cur_scores,
            rate > np.random.rand(self.batch_size)
            )
        acc_smiles = list(np.where(accepted, new_smiles, self.cur_batch))
        acc_scores = np.where(accepted, new_scores, self.cur_scores)
        return acc_scores, acc_smiles
            
    def flush_file(self, scores, smiles):
        with open(self.save_path, 'a') as f:
            for score, smi in zip(scores,smiles):
                f.write("{},{},{}\n"\
                    .format(str(self.step),str(score),smi))

    def __iter__(self):
       return self

    def __next__(self):
        if self.step < self.max_step:
            seeding_z_vecs = self.generate_new_vecs()
            new_scores, self.z_vecs, new_smiles = \
                self.generate_new_smiles(
                    seeding_z_vecs, self.n_samples, 
                    self.oracle, self.sort_descending
                )
            self.cur_scores, self.cur_batch = \
                self.accept_new(new_scores, new_smiles, rate = 0.05) 

            self.step += 1
            if self.save_path is not None:
                self.flush_file(list(self.cur_scores), self.cur_batch)

            torch.cuda.empty_cache()
            return list(self.cur_scores), self.cur_batch
        else:
            raise StopIteration

# class BatchEncoder:
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
    batch_size = 100
    steps = 5
    # n_samples = 10
    torch.manual_seed(42)
    eta = 1
    prms = StockParamLoader(model_loc, vocab_loc)
    test_smi = [
        'C1=CCCCNCCC=CCOCCCC1', # This is not in vocab
        'CCCC',
        'c1ccccc1'
        ]
    test_oracle = lambda x: 0

    with torch.no_grad():
        # Test smiles encoding
        vae = VAEInterface(prms)
        vecs = vae.encode_smiles(test_smi, report_failed=True)
        print("Vecs shape:", vecs.shape)
        torch.cuda.empty_cache()

        start_time = time.time()
        batch_generator = IterGen(
            batch_size, steps, prms, eta, test_oracle,
            'rand'
            )
        for batch in batch_generator:
            print("Step", batch_generator.step)
            print("Total SMILES:", len(batch))
            for i, smi in enumerate(batch):
                print(i, smi)
            print('***'*10)
        print('Time Used:', time.time()-start_time)

        
        start_time = time.time()
        batch_generator = IterGen(
            batch_size, steps, prms, eta, test_oracle,
            'direct'
            )
        for batch in batch_generator:
            print("Step", batch_generator.step)
            print("Total SMILES:", len(batch))
            for i, smi in enumerate(batch):
                print(i, smi)
            print('***'*10)

        print('Time Used:', time.time()-start_time)

        start_time = time.time()
        batch_generator = IterGen(
            batch_size, steps, prms, eta, test_oracle,
            'convert'
            )
        for batch in batch_generator:
            print("Step", batch_generator.step)
            print("Total SMILES:", len(batch))
            for i, smi in enumerate(batch):
                print(i, smi)
            print('***'*10)
        
        print('Time Used:', time.time()-start_time)

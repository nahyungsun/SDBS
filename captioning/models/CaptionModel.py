# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from ..utils import misc as utils
from . import utils as model_utils
import nltk
nltk.download("popular")
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # implements beam search
    # calls beam_step and returns the final set of beams
    # augments log-probabilities with diversity terms when number of groups > 1

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        def hamming_dist(array_a, array_b):  # "" 문자열만 가능
            array_a_len = len(array_a)
            array_b_len = len(array_b)
            if array_a_len < array_b_len:
                array_a += " " * (array_b_len - array_a_len)
            else:
                array_b += " " * (array_a_len - array_b_len)
            count = 0
            for i in range(len(array_a)):
                if array_a[i] == array_b[i]:
                    count += 1
            print("Hamming~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            return count / len(array_a)

        def wordnet_similarity(word1, word2):  # "" 문자열만 가능
            wn_word1 = wn.synsets(word1)
            wn_word2 = wn.synsets(word2)

            if wn_word1 == [] or wn_word2 == []:
                return 0
            print("sdbs~~~~~~~~~~~~~~~~~")
            return wn_word1[0].wup_similarity(wn_word2[0])
            
        def wordnet_syns(word1, word2):  # "" 문자열만 가능
            wn_word1 = wn.synsets(word1)
            if wn_word1 == []:
                return 0
            elif word1 == word2:
                return 1
            elif word2 in wn.synsets(word1)[0].definition():    
                return -1 
            else:
                return 0

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash, topk1, topk2, w_sim, w_apply):
            #  beam_seq_table : 저장된 문장 테이블
            #  logprobs  : 다음 단어 확률 점수
            #  t : 현재 진행 시점
            #  divm : 그룹의 길이
            #  diversity_lambda : 다양성 파라미터
            #  bdash : group당 beam 크기
            #  topk1,2 적용할 단어 개수
            #  w_sim 단어 유사도 기준
            #  w_apply 적용할 단어의 개수

            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]
            
            if divm > 0:
                if local_time == 0:
                    change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                else:
                    change = logprobs.new_zeros(bdash, logprobs.shape[-1])
                # change 에 빼줄 가중치 해주면 됨  다음 그룹의 단어들을 생성할 확률 표 처음에는 (1,vocab) 그 이후는 (beam_size, vocab)
                for prev_choice in range(divm):  # 확률표에 이전 그룹의 단어들에 대한 유사도 가중치 부여
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time] # Nxb 이전 그룹별 단어들 목록 저장(beam_size)
                    for prev_labels in range(bdash):  # beam_size 만큼 반복 이전 그룹의 모든 단어를 보기 위함
                        # change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1), change.new_ones(batch_size, 1))
                        # 기존 diverse라고 해놓은 알고리즘
                        temp = "{}".format(prev_decisions[:, prev_labels].item())  # 이전 그룹의 단어
                        if "{}".format(temp) == '0':  # 0이 나올경우 건너뛰기
                            continue
                        prev_ward = self.vocab[temp]  # 이전 단어 index to string
                        # 이전 단어
                        if local_time == 0:
                            top_k_score, top_k_words = logprobs[0].topk(topk1, 0, True, True)  # 확률표에서 점수가 높은 상위 단어 추출
                            top_k_words_len = len(top_k_words)  # 추출한 단어들의 개수
                            for j in top_k_words:  # 추출한 단어들 하나씩 가져오기
                                temp = "{}".format(j)  # 추출한 단어 index to string
                                if temp == '0':  # 0이 나올경우 건너뛰기
                                    pass
                                else:
                                #hamming_dist
                                #wordnet_similarity
                                    temp_sim = wordnet_similarity(prev_ward, self.vocab[temp])
                                    if temp_sim > w_sim:
                                        change[0][j] += temp_sim / bdash
                                    
                        elif local_time < w_apply:
                            for beam in range(bdash):
                                top_k_score, top_k_words = logprobs[beam].topk(topk2, 0, True, True)  # 확률표에서 점수가 높은 상위 단어 추출
                                top_k_words_len = len(top_k_words)  # 추출한 단어들의 개수
                                for j in top_k_words:  # 추출한 단어들 하나씩 가져오기
                                    temp = "{}".format(j)  # 추출한 단어 index to string
                                    if temp == '0':  # 0이 나올경우 건너뛰기
                                        pass
                                    else:
                                        temp_sim = wordnet_similarity(prev_ward, self.vocab[temp])
                                        if temp_sim > w_sim:
                                            change[beam][j] += temp_sim / bdash
                                        

                logprobs = logprobs - change * diversity_lambda 


                """
            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time] # Nxb
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1), change.new_ones(batch_size, 1))
                
                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda / bdash
                else:
                    logprobs = logprobs - model_utils.repeat_tensors(bdash, change) * diversity_lambda / bdash
                """
                

            return logprobs, unaug_logprobs


        # does one step of classical beam search

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobs: probabilities augmented after diversity N*bxV
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions Nxbxl
            #beam_seq_logprobs : log-probability of each decision made, NxbxlxV
            #beam_logprobs_sum : joint log-probability of each beam Nxb

            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size) # NxbxV
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs # beam_logprobs_sum Nxb logprobs is NxbxV
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:,:beam_size], ix[:,:beam_size]
            beam_ix = ix // vocab_size # Nxb which beam
            selected_ix = ix % vocab_size # Nxb # which world
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1) # N*b which in Nxb beams


            if t > 0:
                # gather according to beam_ix
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) == beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
                
                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(beam_seq_logprobs))
            
            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1) # beam_seq Nxbxl
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1, beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size)) # NxbxV
            assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([
                beam_seq_logprobs,
                beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)
            
            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
            #  copy over state in previous beam q to new beam at vix
                new_state[_ix] = state[_ix][:, state_ix]
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state

        # Start diverse_beam_search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1) # This should not affect beam search, but will affect dbs
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size # beam per group
        topk1 = opt.get('topk1', 0)
        topk2 = opt.get('topk2', 0)
        w_sim = opt.get('sim', 0.8)
        w_apply = opt.get('apply', 15)
        dbs_type = opt.get('dbs_type', 1)
        if dbs_type == 2:
            wordnet_similarity = hamming_dist
            
        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size + 1).to(device) for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        # state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        # state_table = list(zip(*[_.reshape(-1, batch_size * bdash, group_size, *_.shape[2:]).chunk(group_size, 2) for _ in init_state]))
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        # logprobs_table = list(init_logprobs.reshape(batch_size * bdash, group_size, -1).chunk(group_size, 0))
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args = model_utils.split_tensors(group_size, args) # For each arg, turn (Bbg)x... to (Bb)x(g)x...
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)] # group_name, arg_name, model_name
        else:
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]
        step_word_list = []
        for t in range(self.seq_length + group_size - 1):  # 문장의 길이
            for divm in range(group_size):  # 그룹의 길이
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobs = logprobs_table[divm]
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t-divm-1].reshape(-1, 1).to(device), float('-inf'))
                    if remove_bad_endings and t-divm > 0:
                        logprobs[torch.from_numpy(np.isin(beam_seq_table[divm][:, :, t-divm-1].cpu().numpy(), self.bad_endings_ix)).reshape(-1), 0] = float('-inf')
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobs.size(1)-1)] == 'UNK':
                        logprobs[:,logprobs.size(1)-1] = logprobs[:, logprobs.size(1)-1] - 1000
                    elif self.unk_idx is not None:
                        logprobs[:, self.unk_idx] -= 1000
                    # diversity is added here
                    # the function directly modifies the logprobs values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    if group_size > 1 and divm == 0:
                        #print(beam_seq_table)
                        for g_s in range(group_size):
                            #print(beam_seq_table[g_s])
                            for b_s in range(5):
                                #print(beam_seq_table[g_s][0][b_s])
                                if len(beam_seq_table[g_s][0][b_s]) > 0:
                                    step_word_list.append([t, g_s, b_s, len(beam_seq_table[g_s][0][b_s])]+beam_seq_table[g_s][0][b_s].tolist())
                            
                        
                                # t : time step / g_s 순서 / b_s 순서 /문장의 길이/ 테이블
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table,logprobs,t,divm,diversity_lambda,bdash, topk1, topk2, w_sim, w_apply)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm] = beam_step(logprobs,
                                                unaug_logprobs,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t-divm] == self.eos_idx
                        assert beam_seq_table[divm].shape[-1] == t-divm+1
                        if t == self.seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone(), 
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item()
                                }
                                final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                                #print("batch_size : ",b,"bdash : ",vix,"divm : ",divm)
                                #print(final_beam['seq'])
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    # move the current group one step forward in time
                    
                    it = beam_seq_table[divm][:, :, t-divm].reshape(-1).to(logprobs.device)
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it, *(args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # all beams are sorted by their log-probabilities

        #fff=open('Hamming dbs.txt','a')
        #write = csv.writer(fff) 
      
        #write.writerows(done_beams_table)

        #fff.close()
 
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)] for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        #print(done_beams)
        return done_beams

    def old_beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    # local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':unaug_logprobsf[q]})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1) # This should not affect beam search, but will affect dbs
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash, self.vocab_size + 1).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        # state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_state]))
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # Chunk elements in the args
        args = list(args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[_.chunk(group_size) if _ is not None else [None]*group_size for _ in args_] for args_ in args] # arg_name, model_name, group_name
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)] # group_name, arg_name, model_name
        else:
            args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm]
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).to(logprobsf.device), float('-inf'))
                    if remove_bad_endings and t-divm > 0:
                        logprobsf[torch.from_numpy(np.isin(beam_seq_table[divm][t-divm-1].cpu().numpy(), self.bad_endings_ix)), 0] = float('-inf')
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobsf.size(1)-1)] == 'UNK':
                        logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
                    elif self.unk_idx is not None:
                        logprobsf[:, self.unk_idx] -= 1000 
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == self.eos_idx or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(), 
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time
                    
                    it = beam_seq_table[divm][t-divm].to(logprobsf.device)
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it, *(args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = sum(done_beams_table, [])
        return done_beams

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        elif sample_method == 'gumbel': # gumbel softmax
            # ref: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape).to(logprobs.device)
                return -torch.log(-torch.log(U + eps) + eps)
            def gumbel_softmax_sample(logits, temperature):
                y = logits + sample_gumbel(logits.size())
                return F.log_softmax(y / temperature, dim=-1)
            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions
        else:
            logprobs = logprobs / temperature
            if sample_method.startswith('top'): # topk sampling
                top_num = float(sample_method[3:])
                if 0 < top_num < 1:
                    # nucleus sampling from # The Curious Case of Neural Text Degeneration
                    probs = F.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:,:1]), mask[:,:-1]], 1)
                    sorted_probs = sorted_probs * mask.to(sorted_probs)
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())
                else:
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions
        return it, sampleLogprobs


    def decode_sequence(self, seq):
        return utils.decode_sequence(self.vocab, seq)

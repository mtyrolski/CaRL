"""Train small Sokoban BART from scratch on cached frozen-policy labels.

Scratch is the default and reads only the standalone architecture config.
The pretrained mode is retained solely to reproduce the earlier diagnostics.
This bounded experiment runner does not support resuming optimiser state.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BartConfig, BartForConditionalGeneration, BertForSequenceClassification

from scripts.policy_relative_smoke import valid_board, rollout_trials, write_csv
from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer


class ScaleBart(nn.Module):
    def __init__(self, model_source, initialization, condition):
        super().__init__()
        if initialization not in {'scratch','pretrained'}:
            raise ValueError('initialization must be scratch or pretrained')
        config=BartConfig.from_pretrained(model_source,local_files_only=True)
        expected=dict(d_model=256,encoder_layers=3,decoder_layers=3,
            encoder_attention_heads=4,decoder_attention_heads=4,
            encoder_ffn_dim=2048,decoder_ffn_dim=1024)
        assert all(getattr(config,k)==v for k,v in expected.items()),'Use What Matters Table 2 architecture'
        if initialization=='pretrained':
            self.bart=BartForConditionalGeneration.from_pretrained(model_source,local_files_only=True)
            for module in [self.bart.model.encoder,self.bart.model.decoder]:
                original=module.embed_positions
                extended=type(original)(512,config.d_model)
                nn.init.normal_(extended.weight,mean=0,std=config.init_std)
                with torch.no_grad():extended.weight[:len(original.weight)].copy_(original.weight)
                module.embed_positions=extended
            self.bart.config.max_position_embeddings=512
        else:
            config.max_position_embeddings=512
            self.bart=BartForConditionalGeneration(config)
        self.bart.config.use_cache=False
        self.scale_projection=nn.Linear(1,256,bias=False)
        nn.init.zeros_(self.scale_projection.weight)
        self.condition=condition

    def embedding(self,tokens,scale):
        if self.condition=='none':scale=torch.zeros_like(scale)
        return self.bart.model.shared(tokens)+self.scale_projection(scale[:,None])[:,None,:]

    def forward(self,tokens,scale,labels):
        return self.bart(inputs_embeds=self.embedding(tokens,scale),labels=labels,use_cache=False).logits

    @torch.inference_mode()
    def generate(self,tokens,scale):
        self.eval()
        return self.bart.generate(inputs_embeds=self.embedding(tokens,scale),
            max_new_tokens=144,min_new_tokens=144,do_sample=False,num_beams=1,
            use_cache=True,forced_eos_token_id=None,suppress_tokens=list(range(15)))[:,1:]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('analysis/policy_relative_smoke_20261029/main'))
    parser.add_argument('--model-config',type=Path,default=Path('configs/generator/sokoban_bart_small.json'),
                        help='Architecture JSON only; no generator weights are loaded in scratch mode')
    parser.add_argument('--checkpoint',type=Path,default=Path('models_weights/sokoban/generator/checkpoint-6593'),
                        help='Used only with --initialization pretrained (historical diagnostic)')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initialization',choices=['scratch','pretrained'],default='scratch')
    parser.add_argument('--conditions',nargs='+',choices=['relative','absolute','none'],default=['relative','absolute','none'])
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--lr',type=float,default=3e-4)
    args=parser.parse_args()
    if args.steps < 0 or args.batch < 1 or args.eval_every < 1 or args.lr <= 0:
        parser.error('steps must be nonnegative; batch, eval-every and lr must be positive')
    if len(set(args.conditions)) != len(args.conditions):
        parser.error('conditions must not contain duplicates')
    model_source=args.model_config if args.initialization=='scratch' else args.checkpoint
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise FileExistsError('Use a fresh output directory')
    (args.output/'runner_used.py').write_text(Path(__file__).read_text())
    print(json.dumps(dict(initialization=args.initialization,model_source=str(model_source),
                          generator_weights_loaded=args.initialization=='pretrained')),flush=True)
    source=json.loads((args.dataset/'summary.json').read_text());config=source['provenance']['config']
    data=joblib.load(args.dataset/'calibrated_dataset.joblib');rows=data['rows']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    tokenizer=SokobanTokenizer()
    inputs=[];labels=[]
    for s,g,h in zip(data['starts'],data['goals'],data['targets']):
        inputs.append(torch.cat([tokenizer.board_tokenizer(np.eye(7)[s]),torch.tensor([3]),
                                 tokenizer.board_tokenizer(np.eye(7)[g]),torch.tensor([3])]))
        labels.append(tokenizer.board_tokenizer(np.eye(7)[h]))
    tokens=torch.stack(inputs).cuda();targets=torch.stack(labels).cuda()
    weights=torch.tensor([r['weight'] for r in rows],device='cuda')
    masks={s:torch.tensor([r['split']==s for r in rows],device='cuda') for s in ['train','val','test']}
    indices=masks['train'].nonzero().flatten();sampling=weights[indices].cpu()
    selected=[]
    for c in sorted({r['context'] for r in rows if r['split']=='test'}):
        candidates=[i for i,r in enumerate(rows) if r['context']==c]
        selected.extend(sorted({min(candidates,key=lambda i:abs(rows[i]['u']-q)) for q in [.25,.75]}))
    summary=[];all_history=[];raw_proposals={}
    for condition in args.conditions:
        assert condition in ['relative','absolute','none']
        torch.manual_seed(config['seed']);rng=torch.Generator().manual_seed(config['seed'])
        net=ScaleBart(model_source,args.initialization,condition).cuda()
        scales=torch.tensor([r['u'] if condition=='relative' else r['mean_hit_time']/config['cap']
                             if condition=='absolute' else 0 for r in rows],device='cuda',dtype=torch.float32)
        opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();history=[]

        @torch.inference_mode()
        def evaluate(split):
            net.eval();idx=masks[split].nonzero().flatten();losses=[];accuracies=[];changed_ok=[];changed_n=[];exact=[]
            for part in idx.split(args.batch):
                logits=net(tokens[part],scales[part],targets[part]);pred=logits.argmax(-1)
                ce=F.cross_entropy(logits.transpose(1,2),targets[part],reduction='none').mean(1)
                correct=pred==targets[part];changed=tokens[part,:144]!=targets[part]
                losses.append(ce);accuracies.append(correct.float().mean(1))
                changed_ok.append((correct&changed).sum(1));changed_n.append(changed.sum(1));exact.append(correct.all(1).float())
            w=weights[idx]
            return dict(loss=float((torch.cat(losses)*w).sum()/w.sum()),
                teacher_forced_token_accuracy=float((torch.cat(accuracies)*w).sum()/w.sum()),
                teacher_forced_changed_accuracy=float((torch.cat(changed_ok)*w).sum()/(torch.cat(changed_n)*w).sum()),
                teacher_forced_exact_board_accuracy=float((torch.cat(exact)*w).sum()/w.sum()))

        logits: torch.Tensor | None = None
        loss: torch.Tensor | None = None
        for step in range(args.steps+1):
            if step%args.eval_every==0 or step==args.steps:
                metrics: dict[str, str | int | float] = dict(run=f'bart_{args.initialization}_{condition}',step=step)
                for split in masks:metrics.update({split+'_'+k:v for k,v in evaluate(split).items()})
                history.append(metrics);print(json.dumps(metrics),flush=True)
                write_csv(args.output/'learning_curves.csv',all_history+history)
            if step==args.steps:break
            net.train();batch=indices[torch.multinomial(sampling,args.batch,replacement=True,generator=rng).cuda()]
            logits=net(tokens[batch],scales[batch],targets[batch])
            if logits is None:
                raise RuntimeError('BART did not return logits for the training batch')
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets[batch].reshape(-1))
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        train_seconds=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated()/2**20
        generated=[]
        for begin in range(0,len(selected),args.batch):
            chosen=torch.tensor(selected[begin:begin+args.batch],device='cuda')
            output=net.generate(tokens[chosen],scales[chosen]).cpu()
            for sequence in output:
                board=tokenizer.board_detokenizer(sequence.tolist())
                assert board is not None and board.shape==(12,12,7)
                generated.append(board.argmax(-1))
        raw_proposals[condition]=np.stack(generated)
        folder=args.output/condition;folder.mkdir(exist_ok=True)
        net.bart.save_pretrained(folder/'checkpoint')
        torch.save(net.scale_projection.state_dict(),folder/'scale_projection.pt')
        (folder/'trainer_state.json').write_text(json.dumps({'log_history':history},indent=2)+'\n')
        result=dict(condition=condition,initialization=args.initialization,
            parameters=sum(p.numel() for p in net.parameters()),train_seconds=train_seconds,peak_allocated_mib=peak,
            initial=history[0],final=history[-1])
        summary.append(result);all_history.extend(history)
        del opt,net,logits,loss
        gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(args.output/'generated_boards.npz',row_indices=selected,**raw_proposals)
    env=SokobanEnv(tokenizer)
    policy=BertForSequenceClassification.from_pretrained(config['policy'],local_files_only=True)
    cast(nn.Module, policy).cuda()
    policy.eval()
    for result in summary:
        pred=raw_proposals[result['condition']]
        valid=[valid_board(p,data['starts'][i]) for p,i in zip(pred,selected)]
        copies=[np.array_equal(p,data['starts'][i]) for p,i in zip(pred,selected)]
        executable=[j for j in range(len(selected)) if valid[j] and not copies[j]]
        if executable:
            hits,info=rollout_trials(policy,tokenizer,env,
                [np.eye(7,dtype=np.float32)[data['starts'][selected[j]]] for j in executable],
                [np.eye(7,dtype=np.float32)[pred[j]] for j in executable],
                repeats=2,cap=config['cap'],seed=config['seed']+1,batch_size=config['policy_batch'])
        else:hits=np.zeros((0,2));info={'attempted_actions':0}
        result['generation']=dict(requests=len(selected),valid=sum(valid),copies=sum(copies),valid_noncopy=len(executable),
            exact_reference_targets=sum(np.array_equal(p,data['targets'][i]) for p,i in zip(pred,selected)),
            rollout_trials=int(hits.size),successful_noncopy_rollouts=int((hits>0).sum()),**info)
    report=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_calibration=str(args.dataset/'summary.json'),
        seed=config['seed'],model_source=str(model_source),
        generator_weights_loaded=args.initialization=='pretrained',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        architecture='What Matters Table 2: BART d=256, 3+3 layers, 4 heads, FFN=2048/1024',
        adaptation='max positions 256 -> 512 for state + final goal; learned scalar projection; no larger width/depth',
        caveat='Pretrained runs may reuse boards seen in old pretraining; scratch runs hold out layouts from generator training.',runs=summary)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(args.output/'learning_curves.csv',all_history)
    write_csv(args.output/'metrics.csv',[dict(parameters=r['parameters'],train_seconds=r['train_seconds'],
        peak_allocated_mib=r['peak_allocated_mib'],**r['final'],**r['generation']) for r in summary])
    print('DONE',json.dumps(report),flush=True)


if __name__=='__main__':main()

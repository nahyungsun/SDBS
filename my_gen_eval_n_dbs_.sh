#! /bin/sh
set -m
id=$1


for sim in 0
do
for apply in 20
do
for beam in 5
do
for sample in 5
do
for i1 in 50
do
for i2 in 50
do
topk1=$i1
topk2=$i2
for i in 3.0
do


python eval.py --batch_size 1 --diversity_lambda $i --image_root /data/coco/ --dump_images 0 --num_images -1 --split test  --model log_$id/model-best.pth --language_eval 0 --beam_size $beam --sample_n $sample --temperature $3 --sample_method greedy --sample_n_method dbs --infos_path log_$id/infos_$id-best.pkl --id $id"1_hdbs_sim_"${sim}"_apply_"${apply}"_beam_sample_"${beam}_${sample}"_tok1,2_"${topk1}_${topk2}"_diverse_"${i} --topk1 $topk1 --topk2 $topk2 --image_k 500 --sim ${sim} --apply ${apply} --device cpu --dbs_type 2


python eval.py --batch_size 1 --diversity_lambda $i --image_root /data/coco/ --dump_images 0 --num_images -1 --split test  --model log_$id/model-best.pth --only_lang_eval 1 --language_eval 1 --beam_size $beam --sample_n $sample --temperature $3 --sample_method greedy --sample_n_method dbs --infos_path log_$id/infos_$id-best.pkl --id $id"1_hdbs_sim_"${sim}"_apply_"${apply}"_beam_sample_"${beam}_${sample}"_tok1,2_"${topk1}_${topk2}"_diverse_"${i} --device cpu

done
done
done
done
done
done
done







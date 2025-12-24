"""
Usage:
    python /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/batchinf_pr.py \
        test_buckeye \
        anything \
        /work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1 \
        exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000 \
        0
"""
# /work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000
import os
from espnet2.bin.s2t_inference import Speech2Text
import soundfile as sf
from tqdm import tqdm
import sys

testset = sys.argv[1] # name of test set
taskname = sys.argv[2] # name of this setting
basedir = sys.argv[3] # path to exp_name/conf_name dir
expdir = sys.argv[4] # expdir/expname
split = int(sys.argv[5]) # split number, 0-15
if testset in ["test_buckeye", "test_l2artic"]: lang = "<eng>"
else: lang = "<unk>"

readonly_expdir = f"{basedir}/{expdir}"

bpebase = "bpe_unigram40000"
model = Speech2Text(s2t_train_config=f"{expdir}/config.yaml",
                    s2t_model_file=f"{readonly_expdir}/valid.acc.ave_5best.till40epoch.pth", #valid.acc.best.pth",
                    bpemodel=f"{basedir}/data/token_list/{bpebase}/bpe.model", 
                    beam_size=1, ctc_weight=0.3)

with open(f"dump/raw/{testset}/wav.scp",  "r") as f:
    audios = [line.split()[1] for line in f.readlines()]
with open(f"dump/raw/{testset}/text", "r") as f:
    names = [line.split()[0] for line in f.readlines()]

# make 16 splits
size = len(audios)
start, end = split * (size // 16), (split + 1) * (size // 16)
if split==15: end = size  # last split takes the rest

if os.path.exists(f"preds/{testset}/pr-{taskname}-{split}.txt"):
    with open(f"preds/{testset}/pr-{taskname}-{split}.txt", "r") as f:
        skip = sum(1 for _ in f)
else:
    skip = 0
skip += start

# prpred = open(f"preds/{testset}/pr-{taskname}-{split}.txt", "a")

for i in tqdm(range(start, end)):
    if i < skip: continue
    speech, _ = sf.read(audios[i])
    na = "<na>"
    print('Running forward')
    pr = model(speech, text_prev=na, lang_sym=lang, task_sym="<pr>")[-1].states['ctc']
    print(pr)
    print(pr.shape)
    break
    # pr = model(speech, text_prev=na, lang_sym=lang, task_sym="<pr>")[0][0]
    # prpred.write(f"{names[i]} {pr}\n")
    # if i % 10 == 0:
    #     prpred.flush()

# prpred.close()
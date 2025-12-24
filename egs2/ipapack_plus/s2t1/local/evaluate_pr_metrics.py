import panphon.distance

## File loading
testset = "test_fleurs"
task = "pr"
model = "0914"
# change to your output path
with open(f"preds/{testset}/{task}-{model}.txt",  "r") as f:
    # note: split by space if there's no task token
    preds = [line.split(">")[-1].strip() for line in f.readlines()]
# load text.good for pr and g2p tasks
with open(f"dump/raw/{testset}/text.good", "r") as f:
    refs = [line.split(" ")[-1].strip() for line in f.readlines()]

## PFER calculation
import panphon.distance

# formatting: make them phone sequence
cleaner = {"ẽ": "ẽ", "ĩ": "ĩ", "õ": "õ", "ũ": "ũ", # nasal unicode
            "ç": "ç", "g": "ɡ", # common unicode
            "-": "", "'": ""} # noise
def clean(phones):
    cleaned = ""
    for phone in phones: cleaned += cleaner.get(phone, phone)
    return cleaned
hyps = [x.replace("//", "") for x in preds] # for POWSM output
refs = [clean(x) for x in refs] # unicode issue
dst = panphon.distance.Distance()

score = dst.phoneme_error_rate(hyps, refs)
print(f"PER: {score*100:.2f}")

score = dst.feature_error_rate(hyps, refs)
print(f"FER: {score*100:.2f}")
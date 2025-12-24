"""
Usage:

srun -A bbjs-delta-cpu -p cpu --time=2-00:00:00 --ntasks=1 --cpus-per-task=8 \
    bash -lc "source /u/sbharadwaj/conda/etc/profile.d/conda.sh && \
            conda activate powsm2 && python local/prep_buckeye_segmentations.py \
            --input /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/cuts.000000.jsonl \
            --output /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/aligned.cuts.000000.jsonl \
            --buckeye_path_pattern /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye/{SPEAKER}.zip"

"""

import buckeye
import os
import json
from tqdm import tqdm
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


################################################################################################
# Define the buckeyeUnicodeIPArelation and the buckeyeToUnicodeIPA function
buckeyeUnicodeIPArelation = {
    ('a', 'ʌ'),
    ('aa', 'ɑ'),
    ('aan', 'ɑ̃'),
    ('ae', 'æ'),
    ('aen', 'æ̃'),
    ('ah', 'ʌ'),
    ('ahn', 'ʌ̃'),
    ('an', 'ʌ̃'),
    ('ao', 'ɔ'),
    ('aon', 'ɔ̃'),
    ('aw', 'aʊ'),
    ('awn', 'aʊ'),
    ('ay', 'aɪ'),
    ('ayn', 'aɪ'),
    ('b', 'b'),
    ('ch', 'tʃ'),
    ('d', 'd'),
    ('dh', 'ð'),
    ('dx', 'ɾ'),
    ('eh', 'ɛ'),
    ('ehn', 'ɛ̃'),
    ('el', 'l̩'),
    ('em', 'm̩'),
    ('en', 'n̩'),
    ('eng', 'ŋ̩'),
    ('er', 'ə˞'),
    ('ern', 'ə˞'),
    ('ey', 'eɪ'),
    ('eyn', 'eɪ̃'),
    ('f', 'f'),
    ('g', 'g'),
    ('h', 'h'),
    ('hh', 'h'),
    ('hhn', 'h̃'),
    ('i', 'ɪ'),
    ('id', 'ɪ'),
    ('ih', 'ɪ'),
    ('ihn', 'ɪ̃'),
    ('iy', 'i'),
    ('iyih', 'i'),
    ('iyn', 'ɪ̃'),
    ('jh', 'dʒ'),
    ('k', 'k'),
    ('l', 'l'),
    ('m', 'm'),
    ('n', 'n'),
    ('ng', 'ŋ'),
    ('nx', 'ɾ̃'),
    ('ow', 'oʊ'),
    ('own', 'oʊ'),
    ('oy', 'ɔɪ'),
    ('oyn', 'ɔɪ'),
    ('p', 'p'),
    ('q', 'ʔ'),
    ('r', 'ɹ'),
    ('s', 's'),
    ('sh', 'ʃ'),
    ('t', 't'),
    ('th', 'θ'),
    ('tq', 'ʔ'),
    ('uh', 'ʊ'),
    ('uhn', 'ʊ̃'),
    ('uw', 'u'),
    ('uwix', 'u'),
    ('uwn', 'ũ'),
    ('v', 'v'),
    ('w', 'w'),
    ('y', 'j'),
    ('z', 'z'),
    ('zh', 'ʒ')
}

def buckeyeUnicodeToIPA(buckeyeSymbol):
    mapping = dict(buckeyeUnicodeIPArelation)
    return mapping.get(buckeyeSymbol, '')  # Return empty string if not found

class PhonemeAligner:
    def __init__(self, buckeye_path_pattern):
        self.buckeye_path_pattern = buckeye_path_pattern
        self.DELIMITER = ['VOCNOISE', 'NOISE', 'SIL']
        self.FORBIDDEN = ['{B_TRANS}', '{E_TRANS}', '<EXCLUDE-name>', 'LAUGH', 'UNKNOWN', 'IVER-LAUGH', '<exclude-Name>', 'IVER']
        self.speaker_to_buckeye_data = {}

    def load_data_for_speaker(self, speaker_id):
        # load and index data for a speaker from all tracks
        buckeye_data = {}
        speaker_path = self.buckeye_path_pattern.format(SPEAKER=speaker_id)
        if os.path.exists(speaker_path):
            speaker = buckeye.Speaker.from_zip(speaker_path)
            buckeye_data['original'] = speaker
            for track_idx, track in enumerate(speaker.tracks):
                buckeye_data[track_idx] = [(i, word.orthography) for i, word in enumerate(track.words) if word not in self.DELIMITER and word not in self.FORBIDDEN and type(word)==buckeye.containers.Word]
                # we will search the orthographics from jsonl against this list of words
        else:
            print(f"Directory {speaker_path} does not exist.")
        return buckeye_data
    
    def search_orthographic_in_track(self, orthographic, buckeye_data):
        # search for the orthographic in the list of words from a track
        # returns list of tuples (track_idx, [list of indices], similarity)
        orthographic = orthographic.strip().split()
        matches = []
        for k_name, val in buckeye_data.items():
            if k_name == 'original':
                continue
            track_idx = k_name
            track_i = [i for i,_ in val]
            track_w = ' '.join([word.lower().strip() for _,word in val]).split()
            assert len(track_i) == len(track_w), "Length mismatch between indices and words"
            for i in range(len(track_w) - len(orthographic) + 1):
                segment = track_w[i:i+len(orthographic)]
                # edit distance between latin_text and line starting at ith index
                similarity = SequenceMatcher(None, ' '.join(segment), ' '.join(orthographic)).ratio()
                if similarity > 0.8:
                    matches.append((track_idx, track_i[i:i+len(orthographic)], similarity))
        if not matches:
            print("No matches found for:", ' '.join(orthographic))
            return []
        # return exact matches if any
        exact_matches = [m for m in matches if m[2] == 1.0]
        if len(exact_matches) > 0:
            return exact_matches
        # return top matches sorted by similarity
        matches = sorted(matches, key=lambda x: x[2], reverse=True)
        return matches

    def process_jsonl_entry(self, entry):
        # load the data for the speaker
        speaker_id = entry['id'].split('_')[0]
        if speaker_id not in self.speaker_to_buckeye_data:
            self.speaker_to_buckeye_data[speaker_id] = self.load_data_for_speaker(speaker_id)
        buckeye_data = self.speaker_to_buckeye_data[speaker_id]

        # search for the orthographics of entry in text and get track and indices for the matches
        orthographic = entry['supervisions'][0]['custom']['orthographic'].strip()
        assert len(entry['supervisions'])==1, "Multiple supervisions not supported!"
        matches = self.search_orthographic_in_track(orthographic, buckeye_data)
        if not matches:
            entry['alignment'] = []
            return entry
        matches = matches[0]
        if matches[2] < 1.0:
            print(f"Warning: No exact match found for {orthographic} in speaker {speaker_id}")
            entry['alignment'] = []
            return entry
        # add information about the time alignment to json entry
        track_idx, word_indices, similarity = matches
        track = buckeye_data['original'].tracks[track_idx]
        words_in_segment = [track.words[i] for i in word_indices]
        alignment = []
        for word in words_in_segment:
            for phone in word.phones:
                ipa_phone = buckeyeUnicodeToIPA(phone.seg)
                alignment.append([ipa_phone, phone.beg, phone.end])
        entry['alignment'] = alignment
        return entry

    def process_jsonl_file(self, input_file, output_file):
        missing_alignment = 0
        total_phone_alignments = 0
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line_num, line in enumerate(tqdm(infile, desc="Processing entries"), 1):
                entry = json.loads(line.strip())
                processed_entry = self.process_jsonl_entry(entry)
                if len(processed_entry.get('alignment', []))==0:
                    missing_alignment+=1
                total_phone_alignments += len(processed_entry.get('alignment', []))
                outfile.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')
        print(f"Total entries with missing alignment: {missing_alignment}/{line_num}")
        print(f"Total phone alignments: {total_phone_alignments}")


def process_jsonl_file_parallel(input_file, output_file, buckeye_path_pattern, num_processes=8):
    # Load all entries from the input file
    entries = []
    with open(input_file, 'r') as infile:
        for line in infile:
            entries.append(json.loads(line.strip()))
    print(f"Loaded {len(entries)} entries for processing")
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create arguments for each entry (entry, buckeye_path_pattern)
        args = [(entry, buckeye_path_pattern) for entry in entries]
        
        # Process all entries in parallel with progress bar
        results = list(tqdm(
            executor.map(process_entry_worker, args),
            total=len(entries),
            desc="Processing entries"
        ))
    
    missing_alignment = sum(1 for result in results if len(result.get('alignment', [])) == 0)
    total_phone_alignments = sum(len(result.get('alignment', [])) for result in results)
    
    with open(output_file, 'w') as outfile:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Total entries with missing alignment: {missing_alignment}/{len(entries)}")
    print(f"Total phone alignments: {total_phone_alignments}")


def process_entry_worker(args):
    """
    Worker function for processing a single entry in parallel.
    Each process maintains its own PhonemeAligner instance with cache.
    """
    entry, buckeye_path_pattern = args
    aligner = PhonemeAligner(buckeye_path_pattern)
    return aligner.process_jsonl_entry(entry)


def main():
    # Initialize the aligner with the base path to your data
    BUCKEYE_PATH_PATTERN = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye/{SPEAKER}.zip"
    print("Aligner initialized.")
    # Process the JSONL file
    input_file = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/cuts.000000.jsonl"
    output_file = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/aligned.cuts.000000.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Processing {input_file} and saving to {output_file}")

    # single process
    # aligner = PhonemeAligner(BUCKEYE_PATH_PATTERN)
    # aligner.process_jsonl_file(input_file, output_file)
    # multi process
    process_jsonl_file_parallel(input_file, output_file, BUCKEYE_PATH_PATTERN)
    print("Processing complete!")

if __name__ == "__main__":
    main()

################################################################################################

# BUCKEYE_PATH_PATTERN='/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye/{SPEAKER}.zip'

# speaker='s10'

# speaker=buckeye.Speaker.from_zip(BUCKEYE_PATH_PATTERN.format(SPEAKER=speaker))
# print(speaker)

# print(speaker.name, speaker.sex, speaker.age, speaker.interviewer)

# for track in speaker:
#     print(track.name)


# print(speaker.tracks)

# track = speaker.tracks[0]

# print('--------------------')
# print(len(track.words))
# for word in track.words[20:100]:
#     print(word)
#     print(type(word))
# print('--------------------')

# word = track.words[4]

# print(word.orthography)
# print(word.beg)
# print(word.end)
# print(word.dur)
# print(word.phonemic)
# print(word.phonetic)
# print(word.pos)
# print(word.misaligned)

# for phone in word.phones:
#     print(phone.seg, phone.beg, phone.end, phone.dur)

# for phone in track.phones[:10]:
#     print(phone.seg, phone.beg, phone.end, phone.dur)
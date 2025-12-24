import json
import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

ipa_to_arpabet_epitran = {
    "ɑ": "AA",
    "æ": "AE",
    "ʌ": "AH",          # unstressed “uh”; schwa is AX
    "ɔ": "AO",
    "aʊ": "AW",
    "aɪ": "AY",
    "b": "B",
    "tʃ": "CH",
    "d": "D",
    "ð": "DH",
    "ɛ": "EH",
    "ɚ": "AXR",        # r-colored schwa
    "ɝ": "ER",         # stressed r-colored vowel
    "eɪ": "EY",
    "f": "F",
    "ɡ": "G",
    "h": "HH",
    "ɪ": "IH",
    "i": "IY",
    "dʒ": "JH",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "oʊ": "OW",
    "ɔɪ": "OY",
    "p": "P",
    "ɹ": "R",
    "s": "S",
    "ʃ": "SH",
    "t": "T",
    "θ": "TH",
    "ʊ": "UH",
    "u": "UW",
    "v": "V",
    "w": "W",
    "j": "Y",
    "z": "Z",
    "ʒ": "ZH",
    "ə": "AX",
    "ɨ": "IX",
    "l̩": "EL",        # syllabic consonants
    "m̩": "EM",
    "n̩": "EN",
    "ŋ̩": "NX",
    "ɾ̃": "NX",
    "ʔ": "Q",
    'ə˞': 'ER',
    'ɚ': 'ER',
    'ɝ': 'ER',
}

class PhonemeAligner:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.ipa_to_arpabet = self._create_ipa_to_arpabet_mapping()
    
    def _create_ipa_to_arpabet_mapping(self) -> Dict[str, str]:
        """Create comprehensive IPA to ARPABET mapping"""
        return ipa_to_arpabet_epitran
        # return {
        #     # Vowels
        #     'i': 'iy', 'ɪ': 'ih', 'e': 'ey', 'ɛ': 'eh', 'æ': 'ae',
        #     'ɑ': 'aa', 'ɔ': 'ao', 'o': 'ow', 'ʊ': 'uh', 'u': 'uw',
        #     'ʌ': 'ah', 'ə': 'ax', 'ɚ': 'er', 'ɝ': 'er', 'aɪ': 'ay',
        #     'aʊ': 'aw', 'ɔɪ': 'oy', 'oʊ': 'ow', 'eɪ': 'ey',
            
        #     # Consonants
        #     'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
        #     'f': 'f', 'v': 'v', 'θ': 'th', 'ð': 'dh', 's': 's', 'z': 'z',
        #     'ʃ': 'sh', 'ʒ': 'zh', 'h': 'hh', 'm': 'm', 'n': 'n', 'ŋ': 'ng',
        #     'l': 'l', 'r': 'r', 'w': 'w', 'j': 'y', 'tʃ': 'ch', 'dʒ': 'jh',
            
        #     # R-colored vowels
        #     'ə˞': 'er', 'ɚ': 'er', 'ɝ': 'er', 'aʊə˞': 'aw er', 'aɪə˞': 'ay er',
            
        #     # Common variations
        #     'ɾ': 't',  # flapped t
        #     'ʔ': 't',  # glottal stop often maps to t
            
        #     # Stress markers (ignore these)
        #     'ˈ': '', 'ˌ': '', '.': '',
        # }
    
    def convert_ipa_to_arpabet(self, ipa_text: List[str], verbose: bool=False) -> List[str]:
        """Convert IPA text to ARPABET phonemes"""
        # Remove stress markers and word boundaries
        arpabet_phones = []
        i = 0
        
        while i < len(ipa_text):
            found = False
            segment=ipa_text[i]
            if segment in self.ipa_to_arpabet:
                arpabet = self.ipa_to_arpabet[segment]
                if arpabet:  # Skip empty mappings
                    arpabet_phones.extend(arpabet.split())
                    # print('extending with', arpabet.split(), 'for', segment)
                i += 1
                found = True
                continue
    
            if not found:
                # Skip unknown characters
                if verbose:
                    print(f"Unknown IPA segment: {segment}")
                arpabet_phones.append('')  # or some placeholder
                i += 1
        
        return arpabet_phones

    def parse_words_file(self, words_file_path: Path, verbose: bool = False) -> List[Dict]:
        """Parse .words/.lab XLabel-style file into structured dicts."""
        if not words_file_path.exists():
            return []
        
        results=[]
        actual_content=False
        for line in open(words_file_path, 'r'):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                actual_content=True
                continue
            
            parts = line.split(';')
            if len(parts) < 4:
                continue
            if not actual_content:
                continue
            
            # replace multiple spaces to a single space in line
            parts[0] = re.sub(r'\s+', ' ', parts[0])

            time, _, word = parts[0].strip().split(' ', maxsplit=2)
            word = word.strip()
            if word[0]=='<' and word[-1] == '>':
                continue  # Skip noise or non-word entries
            if word[0]=="{" and word[-1] == "}":
                continue  # Skip noise or non-word entries
            time = float(time)
            official_phones = parts[1].strip().split()
            speaker_phones = parts[2].strip().split()

            word = word.lower()
            # Skip empty words
            if not word:
                continue

            results.append({
                "time": time,
                "word": word,
                "official_phones": official_phones,
                "speaker_phones": speaker_phones,
            })
        if verbose:
            print(results[:20])
        return results

    def parse_phones_file(self, phones_file_path: Path, verbose: bool = False) -> List[Dict]:
        """Parse .phones file and extract phone timing info"""
        phones_data = []
        
        if not phones_file_path.exists():
            return phones_data
        
        actual_content=False
        with open(phones_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    actual_content=True
                    continue
                if not actual_content:
                    continue
                    
                parts = line.split()
                if len(parts) >= 3:
                    time = float(parts[0])
                    phone = parts[2]
                    if not phone.islower():  # Skip noise or non-phoneme entries
                        continue
                    if phone[0]== '<' and phone[-1] == '>':
                        continue  # Skip noise or non-phoneme entries
                    
                    phones_data.append({
                        'time': time,
                        'phone': phone
                    })
        if verbose:
            print(phones_data[:20])
        return phones_data

    def find_word_boundaries(self, latin_text: str, words_data: List[Dict], verbose: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """Find start and end times for the Latin text in words data"""
        # Clean Latin text
        latin_words = latin_text.lower().strip().split()

        if not latin_words or not words_data:
            return None, None, None

        for i in range(len(words_data) - len(latin_words) + 1):
            match = True
            for j, latin_word in enumerate(latin_words):
                latin_word = latin_word.strip().lower()
                if i + j >= len(words_data):
                    match = False
                    break
                
                word_data = words_data[i + j]
                file_word = word_data['word'].lower()

                if file_word != latin_word:
                    match = False
                    break
            
            if match:
                start_time = words_data[i]['time']
                if i + len(latin_words) < len(words_data):
                    end_time = words_data[i + len(latin_words)]['time']
                else:
                    if len(words_data) > 1:
                        avg_duration = (words_data[-1]['time'] - words_data[0]['time']) / len(words_data)
                        end_time = words_data[i + len(latin_words) - 1]['time'] + avg_duration
                    else:
                        end_time = start_time + 1.0  # Default 1 second
                if verbose:
                    print(f"Found word boundaries: {start_time} - {end_time}")
                
                arpa_buckeye_phones = []
                for i in range(i, i + len(latin_words)):
                    arpa_buckeye_phones.extend(words_data[i]['speaker_phones'])
                return start_time, end_time, arpa_buckeye_phones
        
        return None, None, None
    
    def align_phones_with_timing(self, arpabet_phones: List[str], phones_data: List[Dict], 
                                start_time: float, end_time: float, verbose: bool = False) -> List[Tuple[float, float]]:
        """Align ARPABET phones with timing data from phones file"""
        alignments = []
        
        if not arpabet_phones:
            return alignments
        eps = 0.2
        # Filter phones data to the time window
        relevant_phones = [p for p in phones_data if start_time - eps <= p['time'] <= end_time + eps]
        
        if not relevant_phones:
            if verbose:
                print("No relevant phones found in the specified time window.")
            alignments = [(-1, -1)] * len(arpabet_phones)
            return alignments
        
        # Try to match phones
        phone_times = [p['time'] for p in relevant_phones]
        phone_labels = [p['phone'].lower() for p in relevant_phones]
        
        # Simple alignment: try to match as many phones as possible
        matched_indices = []
        for arpabet_phone in arpabet_phones:
            best_match_idx = None
            for i, file_phone in enumerate(phone_labels):
                if i not in matched_indices and file_phone == arpabet_phone.lower():
                    best_match_idx = i
                    break
            
            if best_match_idx is not None:
                matched_indices.append(best_match_idx)
            else:
                matched_indices.append(-1)  # No match found
        
        # Create timing alignments
        c=0
        for i, match_idx in enumerate(matched_indices):
            if match_idx != -1 and match_idx < len(phone_times):
                phone_start = phone_times[match_idx]
                if match_idx + 1 < len(phone_times):
                    phone_end = phone_times[match_idx + 1]
                else:
                    phone_end = end_time
                alignments.append((phone_start, phone_end))
            else:
                # No alignment found
                alignments.append((-1, -1))
                c+=1
        
        return alignments, c

    def fuzzy_match_text_in_file(self, latin_text: str, text_file_path: Path) -> bool:
        latin_text = latin_text.lower().strip()
        with open(text_file_path, 'r') as f:
            for line in f:
                line = re.sub(r'<[^>]+> ', '', line)
                line = line.strip().lower().split()
                for i in range(len(line) - len(latin_text.split()) + 1):
                    segment = line[i:i+len(latin_text.split())]
                    # edit distance between latin_text and line strating at ith index
                    similarity = SequenceMatcher(None, ' '.join(segment), latin_text).ratio()
                    if similarity > 0.8:  # Threshold for a match
                        return True
        return False

    def process_jsonl_entry(self, entry: Dict, all_texts: Dict) -> Dict:
        """Process a single JSONL entry and add alignment information"""
        ################################################
        ################################################
        ################################################
        ################################################
        # Extract speaker ID from ID field
        entry_id = entry['id']
        ipa_text = all_texts.get(entry_id, "")
        ipa_text = re.findall(r"[^/]+", ipa_text)
        match = re.match(r's(\d+)_', entry_id)
        if not match:
            print(f"Could not extract speaker ID from {entry_id}")
            return entry
        
        speaker_id = match.group(1)
        recording_id = entry_id.split('-')[0]  # e.g., "s10_4648"
        
        # Find the corresponding files
        speaker_folder = self.base_path / "buckeye" / f"s{speaker_id}"
        # Load all txt files into memory now and search for orthographic text to find the recording session!
        all_text_files = list(speaker_folder.glob("*.txt"))
        cur_text_file=None
        latin_text = entry['supervisions'][0]['custom']['orthographic'].strip()
        # verbose = latin_text == "any holidays at all they just kind of ignore"
        verbose=False
        # print(latin_text)
        for text_file in all_text_files:
            if self.fuzzy_match_text_in_file(latin_text, text_file):
                cur_text_file = text_file
                break
        if cur_text_file is None:
            print(f"Could not find text file for speaker {speaker_id}")
            print(f"Missing text: {latin_text}")
            entry['alignment'] = []
            return entry
        if verbose:
            print(f"Found text file: {cur_text_file} for speaker {speaker_id}")
        ################################################
        ################################################
        ################################################
        ################################################
        # get corresponding phone file to the text file
        recording_id = cur_text_file.stem
        phones_file = speaker_folder / f"{recording_id}.phones"
        words_file = speaker_folder / f"{recording_id}.words"
        
        if not words_file.exists() or not phones_file.exists():
            # print(f"Missing files for {recording_id}")
            entry['alignment'] = []
            return entry
        
        # Parse files
        words_data = self.parse_words_file(words_file, verbose)
        phones_data = self.parse_phones_file(phones_file, verbose)

         # If no valid data, return empty alignment
        
        if len(words_data) == 0 or len(phones_data) == 0:
            print(f"No valid data in files for {recording_id}")
            entry['alignment'] = []
            return entry

        
        start_time, end_time, arpabuckeye_phones = self.find_word_boundaries(latin_text, words_data, verbose)

        if start_time is None or end_time is None:
            print(f"Could not find word boundaries for: {latin_text}, in speaker {speaker_id}")
            entry['alignment'] = []
            return entry
        
        arpapowsm_phones = self.convert_ipa_to_arpabet(ipa_text, verbose)
        if verbose:
            print(len(ipa_text), len(arpabuckeye_phones), '<--- length')
            for a, b in zip(ipa_text, arpabuckeye_phones):
                print(a, b)
        if verbose:
            print(f"IPA Text: {ipa_text}")
            print(f"ARPABET Phones: {arpapowsm_phones}")
            print(f"Start time: {start_time}, End time: {end_time}")
        
        # Align phones with timing
        phone_alignments, missing_count = self.align_phones_with_timing(
            arpapowsm_phones, phones_data, start_time, end_time
        )
        ####################################################
        if verbose:
            print(f"Phone Alignments: {phone_alignments}")
        ####################################################
        
        # Create alignment mapping
        alignment = []        
        # Simple character-to-phone mapping (this could be improved)
        char_idx = 0
        for i, (phone_start, phone_end) in enumerate(phone_alignments):
            if char_idx < len(ipa_text):
                char = ipa_text[char_idx]
                alignment.append((char, [phone_start, phone_end]))
                char_idx += 1
        
        # Handle remaining characters with -1 times
        while char_idx < len(ipa_text):
            char = ipa_text[char_idx]
            alignment.append((char, [-1, -1]))
            char_idx += 1
            missing_count += 1
        
        entry['alignment'] = alignment
        entry['missing_alignment_count'] = missing_count
        return entry
    
    def process_jsonl_file(self, input_file: str, output_file: str):
        """Process entire JSONL file and add alignment information"""
        all_texts = {}
        fpath = self.base_path / 'text'
        with open(fpath, 'r') as all_text:
            for line in all_text:
                line = line.strip()
                element_id, _, text = line.split(' ')
                all_texts[element_id] = text

        missing_alignment=0
        missing_phone_alignments=0
        total_phone_alignments=0
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line_num, line in enumerate(tqdm(infile, desc="Processing entries"), 1):
                entry = json.loads(line.strip())
                processed_entry = self.process_jsonl_entry(entry, all_texts)
                # if line_num > 0:
                #     break
                if len(processed_entry.get('alignment', []))==0:
                    missing_alignment+=1
                missing_phone_alignments += processed_entry.get('missing_alignment_count', 0)
                total_phone_alignments += len(processed_entry.get('alignment', []))
                outfile.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} entries...Missing ratio is {missing_phone_alignments/total_phone_alignments * 100}%, {missing_phone_alignments}/{total_phone_alignments}")
        print(f"Total entries with missing alignment: {missing_alignment}")
        print(f"Total missing phone alignments: {missing_phone_alignments/total_phone_alignments * 100}%, {missing_phone_alignments}/{total_phone_alignments}")

def main():
    # Initialize the aligner with the base path to your data
    base_path = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye"  # Adjust this path
    aligner = PhonemeAligner(base_path)

    # Process the JSONL file
    input_file = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/cuts.000000.jsonl"
    output_file = "/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/alignedlist.cuts.000000.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    aligner.process_jsonl_file(input_file, output_file)
    print("Processing complete!")

if __name__ == "__main__":
    main()
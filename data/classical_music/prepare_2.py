import numpy as np
import pickle
import random
import os

from tokens import *

random.seed(1337)

filepath = "music_dataset"
all_music = []

for root, _, f_names in os.walk(filepath):
    for file in f_names:
        if file.endswith("xml"):
            all_music.append(converter.parse(root+'/'+file).parts[0])

all_songs = []
for part in all_music:
    offset = 0
    error = 0.001
    song_notes = []
    for element in part.flatten():
        if (isinstance(element, note.Note)
            or isinstance(element, chord.Chord)
            or isinstance(element, note.Rest)) \
          and element.offset >= offset - error:
            offset = element.offset + element.quarterLength
            element.offset = 0
            song_notes.append(element)
    all_songs.append(song_notes)

random.shuffle(all_songs)
n = len(all_songs)
train_songs = all_songs[:int(n*0.9)]
val_songs = all_songs[int(n*0.9):]


def transpose_notation(notation, num_semi):
    if isinstance(notation, note.Rest):
        return notation
    return notation.transpose(interval.Interval(num_semi))


def transpose_song(song, num_semi):
    return [transpose_notation(notation, num_semi) for notation in song]


train_songs_transposed = train_songs.copy()
for num_semi in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
    for song in train_songs:
        train_songs_transposed.append(transpose_song(song, num_semi))

train_data = [token for song in train_songs_transposed for token in [None]+tokenize_1(song)]
val_data = [token for song in val_songs for token in [None]+tokenize_1(song)]
all_tokens = train_data+val_data

reduced_tokens = list(set(all_tokens))
vocab_size = len(reduced_tokens)
mtoi = {ch: i for i, ch in enumerate(reduced_tokens)}
itom = {i: ch for i, ch in enumerate(reduced_tokens)}
encode = lambda m: [mtoi[n] for n in m] # encoder: take melody, output a list of integers
decode = lambda l: [itom[i] for i in l] # decoder: take a list of integers, output a melody

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itom': itom,
    'mtoi': mtoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

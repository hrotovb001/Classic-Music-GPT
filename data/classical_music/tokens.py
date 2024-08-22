import math
from music21 import *


def standard_length(num):
    fourth = 4**math.floor(math.log(num, 4))
    return round(num/fourth) * fourth


class NoteToken:
    def __init__(self, pitches, num):
        self.pitches = sorted(pitches)
        self.duration = standard_length(num)

    def __eq__(self, other):
        return isinstance(other, NoteToken) and self.pitches == other.pitches and self.duration == other.duration

    def __hash__(self):
        return hash(" ".join(self.pitches) + " " + str(self.duration))


class PitchToken:
    def __init__(self, pitch_str):
        self.pitch = pitch_str

    def __eq__(self, other):
        return isinstance(other, PitchToken) and self.pitch == other.pitch

    def __hash__(self):
        return hash(self.pitch)


class RestToken:
    def __init__(self, num):
        self.duration = standard_length(num)

    def __eq__(self, other):
        return isinstance(other, RestToken) and self.duration == other.duration

    def __hash__(self):
        return hash(self.duration)


def tokenize_1(music):
    tokens = []
    for music_note in music:
        if music_note.quarterLength <= 0:
            continue
        if isinstance(music_note, note.Note):
            tokens.append(NoteToken([str(music_note.pitch)], music_note.quarterLength))
        if isinstance(music_note, chord.Chord):
            pitches = [str(music_pitch) for music_pitch in music_note.pitches]
            tokens.append(NoteToken(pitches, music_note.quarterLength))
        if isinstance(music_note, note.Rest):
            tokens.append(RestToken(music_note.quarterLength))
    return tokens


def tokenize_2(music):
    tokens = []
    for music_note in music:
        if music_note.quarterLength <= 0:
            continue
        if isinstance(music_note, note.Note):
            tokens.append(PitchToken(str(music_note.pitch)))
        if isinstance(music_note, chord.Chord):
            for note_pitch in music_note.pitches:
                tokens.append(PitchToken(str(note_pitch)))
        tokens.append(RestToken(music_note.quarterLength))
    return tokens


def pitch_list_to_note(pitches, note_duration):
    if len(pitches) == 0:
        return note.Rest(note_duration)
    elif len(pitches) == 1:
        note_ret = note.Note(pitches[0])
        note_ret.quarterLength = note_duration
        return note_ret
    else:
        note_ret = chord.Chord(' '.join(pitches))
        note_ret.quarterLength = note_duration
        return note_ret


def detokenize_1(tokens):
    music = []
    for token in tokens:
        if isinstance(token, NoteToken):
            music.append(pitch_list_to_note(token.pitches, token.duration))
        elif isinstance(token, RestToken):
            music.append(note.Rest(token.duration))
        else:
            music.append(note.Rest(8))
    return music


def detokenize_2(tokens):
    music = []
    chord_notes = []
    for token in tokens:
        if isinstance(token, PitchToken):
            chord_notes.append(token.pitch)
        elif isinstance(token, RestToken):
            music.append(pitch_list_to_note(chord_notes, token.duration))
            chord_notes = []
        else:
            music.append(note.Rest(8))
            chord_notes = []
    return music

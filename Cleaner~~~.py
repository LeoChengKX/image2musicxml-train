import os

import pandas as pd
import numpy as np


SPLITS_FOLDER = '/Users/aaronsu/Desktop/brrrrr/image2musicxml-train/splits'

pieces = {}
for filename in os.listdir(SPLITS_FOLDER):
    pieces[filename] = pd.read_csv(os.path.join(SPLITS_FOLDER, filename))


def faulty(filename: str) -> str:
    faulty_pages = []
    if pieces[filename].loc[0, 'measure'] != 1:
        print(f"{filename} does not start at measure 1")
        faulty_pages.append(0)
    for i in range(len(pieces[filename])):
        if str(pieces[filename].loc[i, 'measure'])[0] == "[" and i != 0:
            print(f"{pieces[filename].loc[i, 'filename']} is list")
            faulty_pages.append(i)
        elif str(pieces[filename].loc[i, 'measure']) == "1" and i != 0:
            print(f"{pieces[filename].loc[i, 'filename']} starts at 1")
            faulty_pages.append(i)
        # elif ("3" in str(pieces[filename].loc[i, 'measure']) or \
        #       "8" in str(pieces[filename].loc[i, 'measure'])) and i != 0:
        #     print(f"{pieces[filename].loc[i, 'filename']} might display 3 and 8 issues")
        #     faulty_pages.append(i)
    return faulty_pages


def check_faulty_piece(pieces: dict) -> dict:
    faulty_pieces = {}
    for piece in pieces:
        faulty_pieces[piece] = faulty(piece)
    for piece in faulty_pieces:
        if len(faulty_pieces[piece]) == len(pieces[piece]):
            faulty_pieces[piece] = "ALL"
        if len(faulty_pieces[piece]) == 0:
            faulty_pieces[piece] = "None"
    return pd.DataFrame(faulty_pieces.items(), columns=['Piece', 'Faulty_Pages'])\
        .sort_values(by="Piece", key=lambda col: col.str.lower(), ignore_index=True)

print(check_faulty_piece(pieces))
        
            



        

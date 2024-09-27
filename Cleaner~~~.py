import os

import pandas as pd
import numpy as np


SPLITS_FOLDER = 'splits'

### Dictionary with piece name as key and dataframe of pages and labeled starting measures as values
pieces = {}
for filename in os.listdir(SPLITS_FOLDER):
    pieces[filename] = pd.read_csv(os.path.join(SPLITS_FOLDER, filename))



### Helper to assign an error number
def issue_no(row: int, filename: str) -> int:
    if row == 0:
        if str(pieces[filename].loc[0, 'measure']) != "1":
            return 0
    if str(pieces[filename].loc[row, 'measure'])[0] == "[" and row != 0:
        return 1
    elif str(pieces[filename].loc[row, 'measure']) == "1" and row != 0:
        return 2
    # elif ("3" in str(pieces[filename].loc[i, 'measure']) or \
    #       "8" in str(pieces[filename].loc[i, 'measure'])) and i != 0:
    #     return 3
    #     print(f"{pieces[filename].loc[i, 'filename']} might display 3 and 8 issues")


### Helper to identify issue in words
def issue_reason(page: int, filename: str) -> str:
    if issue_no(page, filename) == 0:
        return "First measure Error"
    if issue_no(page, filename) == 1:
        return "List Error"
    if issue_no(page, filename) == 2:
        return "1 Error"
    # if issue_no(row, filename) == 3:
    #     return "3/8 Error"



### Generate list of faulty pages for a piece
def faulty_page_list(filename: str) -> str:
    faulty_pages = []
    for i in range(len(pieces[filename])):
        if issue_no(i, filename) == 0:
            faulty_pages.append(0)
        if issue_no(i, filename) == 1:
            faulty_pages.append(i)
        if issue_no(i, filename) == 2:
            faulty_pages.append(i)
        if issue_no(i, filename) == 3:
            faulty_pages.append(i)
    return faulty_pages


### Generate dataframe of faulty pieces and their specific faulty pages
def gen_faulty_df(pieces: dict) -> dict:
    faulty_pieces = {}
    for piece in pieces:
        faulty_pieces[piece] = faulty_page_list(piece)
    for piece in faulty_pieces:
        if len(faulty_pieces[piece]) >= len(pieces[piece]) - 5:
            faulty_pieces[piece] = "FULL CHECK NEEDED"
        if len(faulty_pieces[piece]) == 0:
            faulty_pieces[piece] = "None"
    return pd.DataFrame(faulty_pieces.items(), columns=['Piece', 'Faulty_Pages'])\
        .sort_values(by="Piece", key=lambda col: col.str.lower(), ignore_index=True)

faulty_df = gen_faulty_df(pieces)
print(faulty_df)



### Helper to show current label of a page 
def current_value(piece_no: int, page_no: int):
    return pieces[faulty_df.loc[piece_no, "Piece"]].loc[page_no, "measure"]


### Prints faulty pages and their respective labelling errors
def faulty_pages(piece_no: int) -> list:
    print(faulty_df.loc[piece_no, "Faulty_Pages"])
    if str(faulty_df.loc[piece_no, "Faulty_Pages"]).startswith("[") == True:
        for page in faulty_df.loc[piece_no, "Faulty_Pages"]:
            piece_page = pieces[faulty_df.loc[piece_no, "Piece"]].loc[page, "filename"]
            error = issue_reason(page, faulty_df.loc[piece_no, "Piece"])
            print(f"{piece_page}: [{error}], current value: {current_value(piece_no, page)}")


### Makes changes to starting measure label
def change_measure(piece_no: int, page_no: int, measure_change: int) -> None:
    change_target = faulty_df.loc[piece_no, "Piece"]
    change_df = pd.read_csv(os.path.join(SPLITS_FOLDER, change_target))
    change_df.loc[page_no, "measure"] = measure_change
    change_df["measure"] = change_df["measure"].astype(str)
    change_df.to_csv(os.path.join(SPLITS_FOLDER, change_target), index=False)

change_measure()

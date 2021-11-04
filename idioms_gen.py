"""
# Tosin Adewumi

"""

import csv
import nltk


#DIRECTORY = "data"
RD_FILE = "idiomsAc.csv"
WR_FILE = "target_corpus.csv"


if __name__ == '__main__':
    id_col = ""
    meaning_col = ""
    class_label = ""
    idiom_literal = ""
    token_len, i = 0, 0
    with open(RD_FILE, 'r') as file:
        reader = csv.reader(file)
        row_bool = True
        word_bool = True
        for row in reader:                                      # Read new row
            if row_bool:
                row_bool = False                                # change flag after ignoring col headers
            else:
                list_of_words = nltk.word_tokenize(row[1])      # 2nd (entries) column
                for word in list_of_words:
                    if word_bool:
                        id_col = row[0]                         # 1st (ID) column
                        meaning_col = row[2]                    # 3rd (meaning) column later 4th
                        class_label = row[3]                    # 4th (class label)
                        idiom_literal = row[4]                  # 5th (idiom literal)
                        word_bool = False
                        # PoS Tagging
                        try:
                            # tokenized = nltk.word_tokenize(row[1])
                            tagged = nltk.pos_tag(list_of_words)
                        except Exception as e:
                            print(str(e))
                            # continue
                    else:
                        id_col = ""
                        meaning_col = ""
                        class_label = ""
                        idiom_literal = ""
                    with open(WR_FILE, 'a+', newline='') as file:     # not the most efficient loop
                        writer = csv.writer(file)
                        writer.writerow([id_col, tagged[i][0], tagged[i][1], class_label, meaning_col, idiom_literal]) # token | PoS
                    i += 1                                            # next token in sentence
                word_bool = True                        # change flag when picking new row (record 1st ID column only once)
                i = 0                                   # reset row position for new entry

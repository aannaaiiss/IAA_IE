from sklearn.metrics import cohen_kappa_score
import pandas as pd

#######

filename_annotations1 = "../../BEKOLO_annotations_1-10.txt"
filename_annotations2 = "../../BEKOLO_annotations_1-10.2.txt"

#######

def extract_annotations(filename):
    '''Returns a list of lines contained in the file'''
    
    with open(filename,"r") as f: annotations = f.readlines()
    return [annotation.strip() for annotation in annotations]

def format_data(raw_lines):
    """Returns the list of annotated sentences and a list of list of tuplet from the raw_data"""
    
    sentences = []
    data = []
    i=-1

    for line in raw_lines:
        if "#" in line :
            sentences.append(line.lower())
            data.append([])
            i+=1
        else :
            data[i].append(tuple(line.lower().split("\t")))
    return sentences, data

def calculate_cohens_kappa(annotations1, annotations2):
    """Calculate Cohen's Kappa score """
    
    combined_annotations = list(set(annotations1).union(set(annotations2)))
 
    matrix_annotator1 = [1 if triplet in annotations1 else 0 for triplet in combined_annotations]
    matrix_annotator2 = [1 if triplet in annotations2 else 0 for triplet in combined_annotations]

    kappa_score = cohen_kappa_score(matrix_annotator1, matrix_annotator2)

    return kappa_score

def get_cohen_kappa_df(filename_annotations1,filename_annotations2):
    
    sentences1,annotations1 = format_data(extract_annotations(filename_annotations1))
    sentences2,annotations2 = format_data(extract_annotations(filename_annotations2))
    
    if not len(sentences1) == len(sentences2) : return "unmatched number of sentences"
    
    kappa_scores = [calculate_cohens_kappa(annotation1, annotation2) for annotation1,annotation2 in zip(annotations1,annotations2)]
    kappa_scores += [calculate_cohens_kappa(
        [triplet for sentence in annotations1 for triplet in sentence], 
        [triplet for sentence in annotations2 for triplet in sentence])
                     ]
    
    sentences1 += ["whole data"]

    df = pd.DataFrame({"sentence":sentences1,"Cohen's Kappa score":kappa_scores})
    
    return df
    
######
df = get_cohen_kappa_df(filename_annotations1,filename_annotations2)
df.to_csv("kappa_scores.csv")


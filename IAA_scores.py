from sklearn.metrics import cohen_kappa_score
import pandas as pd

#######
path = '../../'
filename_annotations1 = "BEKOLO_annotations_1-10.txt"
filename_annotations2 = "BEKOLO_annotations_1-10.2.txt"

#######

def extract_annotations(file_path):
    '''Returns a list of lines contained in the file'''
    
    with open(file_path,"r") as f: annotations = f.readlines()
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
    """Returns a df with kappa scores for :
        -each sentence
        -all of the sentences as a whole
        -arguments 1
        -relations
        -arguments 2"""
    
    sentences1,annotations1 = format_data(extract_annotations(filename_annotations1))
    sentences2,annotations2 = format_data(extract_annotations(filename_annotations2))
    
    if not len(sentences1) == len(sentences2) : return "unmatched number of sentences"
    
    #1 Calculate Cohen's Kappa scores for each sentence
    kappa_scores = [calculate_cohens_kappa(annotation1, annotation2) for annotation1,annotation2 in zip(annotations1,annotations2)]
    
    #2 Calculate Cohen's Kappa scores for the whole set of sentences
    whole_annotations1 =  [triplet for sentence in annotations1 for triplet in sentence]
    whole_annotations2 = [triplet for sentence in annotations2 for triplet in sentence]
   
    kappa_scores += [calculate_cohens_kappa(whole_annotations1,whole_annotations2)]
    sentences1 += ["whole data"]
    
     #3 Calculate Cohen's Kappa scores for each element of the triplet
    for i,triplet_el in enumerate(["arg1","relation"]) :
        kappa_scores += [calculate_cohens_kappa(
            [triplet[i] for triplet in  whole_annotations1], 
            [triplet[i] for triplet in  whole_annotations2])
                        ]
        sentences1 += [triplet_el]
        
    kappa_scores += [calculate_cohens_kappa(
            [triplet[2] for triplet in  whole_annotations1 if len(triplet)>2 ], 
            [triplet[2] for triplet in  whole_annotations2 if len(triplet)>2])
                        ]
    sentences1 += ["arg2"]
    
    df = pd.DataFrame({"sentence":sentences1,"Cohen's Kappa score":kappa_scores})
    
    return df
    
######
df = get_cohen_kappa_df(path+filename_annotations1,path+filename_annotations2)
df.to_csv(f"{filename_annotations1}_{filename_annotations2}_kappa_scores.csv")


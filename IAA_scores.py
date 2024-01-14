from sklearn.metrics import cohen_kappa_score
import pandas as pd
from collections import defaultdict
from itertools import combinations

######################################################################################################
path = '../../annotations/'
annotators = ["anais","blandine","mathilde"]
first_annotations = "1-10_annotations"
complete_annotations = "1-40_annotations"
######################################################################################################
'''Modify here'''
save_df = False
######################################################################################################

#1 : EXTRACT DATA
def extract_annotations(file_path):
    '''Returns a list of lines contained in the file'''  
      
    with open(file_path,"r",encoding="UTF-8") as f: annotations = f.readlines()
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
            triplet = line.lower().split(" <tab> ")
            triplet = tuple([element.strip() for element in triplet])
            data[i].append(triplet)
    return sentences, data

######################################################################################################

#2: METHOD 1 : COHEN'S KAPPA SCORES
def calculate_cohens_kappa(annotations1, annotations2):
    """Calculate Cohen's Kappa score """
    
    combined_annotations = list(set(annotations1).union(set(annotations2)))
 
    matrix_annotator1 = [1 if triplet in annotations1 else 0 for triplet in combined_annotations]
    matrix_annotator2 = [1 if triplet in annotations2 else 0 for triplet in combined_annotations]

    kappa_score = cohen_kappa_score(matrix_annotator1, matrix_annotator2)

    return kappa_score

def get_cohen_kappa_df(annotations1,annotations2,sentences):
    """Returns a df with kappa scores for :
        -each sentence
        -all of the sentences as a whole
        -arguments 1
        -relations
        -arguments 2
        and if there are any :
        -arguments 3
        -arguments 4
        -arguments 5"""

    #1 Calculate Cohen's Kappa scores for each sentence
    kappa_scores = [calculate_cohens_kappa(annotation1, annotation2) for annotation1,annotation2 in zip(annotations1,annotations2)]
    
    #2 Calculate Cohen's Kappa scores for the whole set of sentences
    whole_annotations1 =  [triplet for sentence in annotations1 for triplet in sentence]
    whole_annotations2 = [triplet for sentence in annotations2 for triplet in sentence]
 
    kappa_scores += [round(calculate_cohens_kappa(whole_annotations1,whole_annotations2),2)]
    sentences += ["whole data"]
    
    for triplet in whole_annotations2:
        if len(triplet)<2:print(triplet)
    
     #3 Calculate Cohen's Kappa scores for each element of the triplet
    for i,triplet_el in enumerate(["arg1","relation"]) :
        kappa_scores += [calculate_cohens_kappa(
            [triplet[i] for triplet in  whole_annotations1], 
            [triplet[i] for triplet in  whole_annotations2])
                        ]
        sentences += [triplet_el]
    
    for i,triplet_el in enumerate(["arg2","arg3","arg4","arg5"]) :
        kappa_scores += [calculate_cohens_kappa(
                [triplet[i+2] for triplet in  whole_annotations1 if len(triplet)>(i+2)], 
                [triplet[i+2] for triplet in  whole_annotations2 if len(triplet)>(i+2)])
                            ]
        sentences += [triplet_el]
    
    df = pd.DataFrame({"sentence":sentences,"Cohen's Kappa score":kappa_scores})
    
    return df

######################################################################################################
#2 : METHOD 2 : AGREEMENT POURCENTAGE
def classify(annotation):
    """Given an annotation, return a nested dictionnary in which you have each token classification according to the triplet
    example : 
    input : ('avram noam chomsky', '[was born in]', 'december 7, 1928') 
    output : {'avram': 0,'noam': 0,'chomsky': 0,'was': 1,'born': 1,'in': 1,'december': 2,'7,': 2,'1928': 2}"""
    classif = {}
    class_index = 0

    for element in annotation:
        for word in element.split():
            word = word.strip("[]")
            classif.setdefault(word, class_index)
        class_index += 1

    return classif

def count_classification(sentence_classif):
    """Given sentence classification of its token, return the number of times a token is classified in a given category
    example :
    input : [
        {'avram': 0, 'noam': 0, 'chomsky': 0, 'was': 1, 'born': 1, 'in': 1, 'december': 2, '7,': 2, '1928': 2},
        {'avram': 0, 'noam': 0, 'chomsky': 0, 'is': 1, 'american': 2}
    ]
    output : {
        'avram': defaultdict(<class 'int'>, {0: 2}), 
        'noam': defaultdict(<class 'int'>, {0: 2}), 
        'chomsky': defaultdict(<class 'int'>, {0: 2}), 
        'was': defaultdict(<class 'int'>, {1: 1}), 
        'born': defaultdict(<class 'int'>, {1: 1}), 
        'in': defaultdict(<class 'int'>, {1: 1}), 
        'december': defaultdict(<class 'int'>, {2: 1}), 
        '7,': defaultdict(<class 'int'>, {2: 1}), 
        '1928': defaultdict(<class 'int'>, {2: 1}), 
        'is': defaultdict(<class 'int'>, {1: 1}), 
        'american': defaultdict(<class 'int'>, {2: 1})
    }"""
    
    counts_dict = defaultdict(lambda: defaultdict(int))

    for triplet_classif in sentence_classif:
        for key, value in triplet_classif.items():
            counts_dict[key][value] += 1

    counts_dict = dict(counts_dict)

    return counts_dict

def count_agreements(classif1,classif2):
    """Given the classification count for a sentence for both annotators, return the number of agreements and the number of possible tokens"""
    common_elements = set(classif1.keys()) & set(classif2.keys())

    total_agreed_count = sum(
        min(classif1[element].get(position, 0), classif2[element].get(position, 0))
        for element in common_elements
        for position in set(classif1.get(element, {}).keys()) & set(classif2.get(element, {}).keys())
    )

    total_count = sum(
        max(classif1[element].get(position, 0), classif2[element].get(position, 0))
        for element in common_elements
        for position in set(classif1.get(element, {}).keys()) | set(classif2.get(element, {}).keys())
    )

    total_count += sum(
        sum(classif1.get(element, {}).values())
        for element in set(classif1.keys()) - common_elements
    )

    total_count += sum(
        sum(classif2.get(element, {}).values())
        for element in set(classif2.keys()) - common_elements
    )
    return total_agreed_count, total_count

def count_agreement_pourcentage(annotations1,annotations2):
    """Return the pourcentage of agreement"""
    total_agreed_count = 0
    total_count = 0

    classif1 = [[classify(annotation) for annotation in sentence] for sentence in annotations1]
    classif2 = [[classify(annotation) for annotation in sentence] for sentence in annotations2]

    count_classif1 = [count_classification(classification) for classification in classif1]
    count_classif2 = [count_classification(classification) for classification in classif2]

    for count1, count2 in zip(count_classif1,count_classif2) :
        agreed_count, max_count = count_agreements(count1,count2)
        total_agreed_count +=agreed_count
        total_count += max_count

    return round(total_agreed_count/total_count,2)

def compute_IAA(filename1,filename2,path,save_df=False):
    sentences,annotations1 = format_data(extract_annotations(path+filename1))
    sentences,annotations2 = format_data(extract_annotations(path+filename2))
    df = get_cohen_kappa_df(annotations1,annotations2,sentences)
    df = df.sort_values(by='sentence', ascending=False)
    df['Agreement pourcentage'] = [count_agreement_pourcentage(annotations1,annotations2)] + [None] * (len(df) - 1)
    if save_df : df.to_csv(f"../../IAA_scores/{filename1}_{filename2}_IAA_scores.csv")
    return df

######################################################################################################
couples =  list(combinations(annotators, 2))

#IAA between first and second versions
for annotator in annotators :
    print(f"IAA between first and second versions : {annotator}")
    print(compute_IAA(f"{first_annotations}_{annotator}.txt",f"{first_annotations}_v2_{annotator}.txt",path,save_df=save_df).iloc[0])
    print()

#IAA between first versions
for annotator1,annotator2 in couples :
    print(f"IAA between first versions : {annotator1} and {annotator2}")
    print(compute_IAA(f"{first_annotations}_{annotator1}.txt",f"{first_annotations}_{annotator2}.txt",path,save_df=save_df).iloc[0])
    print()

#IAA on the complete corpus
for annotator1,annotator2 in couples :
    print(f"IAA on the complete corpus : {annotator1} and {annotator2}")
    print(compute_IAA(f"{complete_annotations}_{annotator1}.txt",f"{complete_annotations}_{annotator2}.txt",path,save_df=save_df).iloc[0])
    print()
    
#IAA between complete corpus and adjucated version
for annotator in annotators :
    print(f"IAA between complete corpus and adjucated version : {annotator}")
    print(compute_IAA(f"{complete_annotations}_{annotator}.txt",f"{complete_annotations}_adjucated.txt",path,save_df=save_df).iloc[0])
    print()





#%%
from dbpedia_query_service import DBpediaQueryService 


service = DBpediaQueryService(vector_file="/Users/chenzichu/Desktop/NED_with_Knowledge_Graph/model.kv")

# Some using case of knowledge graph
# service.get_similarity(concept_1 = "Pink Floyd", concept_2 = "Wall")
# service.find_closest_lemmas(lemma = "Floyd", top = 5)

# %%
# spacy NER
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
# %%
# Exhaustion: Get all the combination
import itertools
import random

def disambiguation(entities, limit = True):
    # If there's a limitation, randomly choose 2 elements
    if limit:
        entities = [random.sample(e, k = 2) for e in entities]

    result = []
    for combination in itertools.product(*entities):
        print(combination)
        score = 0
        for g in list(itertools.combinations(combination,2)):
            try:
                score += service.get_similarity(concept_1 = g[0], concept_2 = g[1])
            except:
                score += 0 
                
        # print(score)
        result.append([combination,score])
    # rank all the result and get the best score
    result = sorted(result, key = lambda r: r[1], reverse = True)
    best_result = result[0]
    return best_result

#%%
# Greedy: get local maximum
def greedy1(entities, early_stop = 0.3):
    q = [[0] * len(entities)]
    score = early_stop * len(entities)

    # Get the score of a list of entities
    # Query 2^len(es) times
    def __score(es):
        s = 0
        for g in list(itertools.combinations(es,2)):
            try:
                s += service.get_similarity(concept_1 = g[0], concept_2 = g[1])
            except:
                s += 0 
        return s

    best_score = __score(entities[k][0] for k in len(entities))
    # Main Loop
    for idx in q:
        for i in range(len(entities)):
            new_idx = idx
            new_idx[i] += 1
            c = tuple([entities[j1][j2] for j1, j2 in enumerate(new_idx)])
            new_score = __score(c)
            if new_score > best_score:
                q.append(new_idx)
                best_score = new_score

            if best_score >= score:
                res_entities = [entities[i1][i2] for i1, i2 in enumerate(q[-1])]
                res_score = best_score
                return [res_entities, res_score]

    return "Requirement not satisfied"     

#%%
def greedy2(entities, early_stop = 0.3):
    d = dict()
    score = early_stop * len(entities)

    for combination in itertools.product(*entities):
        d[combination] = 0

    def __score(es):
        s = 0
        for g in list(itertools.combinations(es,2)):
            try:
                s += service.get_similarity(concept_1 = g[0], concept_2 = g[1])
            except:
                s += 0 
        return s

    for combination in d:
        c = tuple(combination)
        s = __score(c)
        if s >= score:
            return [combination, s]


#%%
entities = [['Floyd Lowa', 'Pink Floyd'],['The Rock','Rock Music'],['Berlin Wall','The Wall']]

disambiguation(entities)

    
#%%
greedy1(entities, early_stop = 0.5)   

# %%
greedy2(entities, early_stop = 0.5)  

import json
from transformers import AutoTokenizer
import spacy
import scispacy
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from contextlib import closing

GLOBALLOCK = mp.Lock()
DATA_PATH = "data/olida.json"
OUTPUT_DIRECTORY = "data/tagged_text/"
MAX_DISTANCE = 512
ANNOTATION_DICT = {
        'Gene': '@GENE$',
        'ProteinMutation': '@VARIANT$',
        'DNAMutation':'@VARIANT$',
        'SNP':'@VARIANT$'
    }
TOKENIZERS_PARALLELISM=True

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
nlp = spacy.load('en_core_sci_md')
s = open(DATA_PATH).read()
data = json.loads(s)


class EntityIterator(object):
    def __init__(self, entity):
        self.entity = entity
        self.index = 0
        
    def __next__(self):
        if self.index < len(self.entity):
            result = self.entity[self.index]
            self.index += 1
            return result
        raise StopIteration
        

class Entity(object):
    def __init__(self, entity_type, mention, identifier, location):
        self.positions = [location]
        self.mentions = [mention]
        self.identifier = identifier
        self.type = entity_type
    
    def __eq__(self, other):
        return (self.identifier == other.identifier) and (self.type == other.type)
    
    def merge(self, other):
        self.positions.extend(other.positions)
        self.mentions.extend(other.mentions)
        positions = np.array(self.positions, dtype=[('start', int), ('end', int)])
        mentions = np.array(self.mentions)
        inds = positions.argsort(order='start')
        self.positions = list(map(tuple, positions[inds]))
        self.mentions = mentions[inds].tolist()
    
    def __repr__(self):
        return f"Entity type {self.type} found at {self.positions} with the text {self.mentions}"
    
    def __str__(self):
        return repr(self)
    
    def get_positions(self):
        if len(self.positions) == 1:
            return self.positions[0]
        return self.positions
    
    def get_start(self):
        return self.positions[0][0]
    
    def get_mentions(self):
        if len(self.mentions) == 1:
            return self.mentions[0]
        return self.mentions
    
    def __len__(self):
        return len(self.positions)
    
    def __iter__(self):
        return EntityIterator(self)
    
    def __lt__(self, other):
        if len(self) > 1:
            raise IndexError("Impossible to sort entities if they are at more than one place")
        return self.positions[0][0] < other.positions[0][0]
        
    
    def __getitem__(self, indx):
        if indx >= len(self.positions):
            raise IndexError("Indice plus grand que la taille de la liste contenant les infos")
        return Entity(self.type, self.mentions[indx], self.identifier, self.positions[indx])
    
class OligoInstance(object):
    tokenizer = tokenizer
    nlp = nlp
    MAX_TOKENS = MAX_DISTANCE
    ENTITY_TAGS = ['@GENE$', '@VARIANT$']
    ENTITY_TAGS_TOKENIZED = [tokenizer.tokenize(tag) for tag in ENTITY_TAGS]
    
    def __init__(self,pmcid, group, text):
        self.pmcid = pmcid
        self.gene1, self.gene2, self.variant1, self.variant2 = self.assign_entities(group)
        self.text, self.tokens, self.length = self.check(text)
    
    @classmethod
    def generate_tokens(self, text):
        return self.tokenizer.tokenize(text)
    
    def is_valid(self):
        return self.length <= self.MAX_TOKENS - 2 # -2 to take into account the [CLS] and [SEP] added by the berttokenizer 
        
    @staticmethod
    def assign_entities(group):
        genes = [entity for entity in group if entity.type == "Gene"]
        variants = [entity for entity in group if entity.type != "Gene"]
        return genes + variants
    
    def __str__(self):
        return f"{self.pmcid}\t{self.gene1.mentions[0]};{self.gene1.identifier}\t{self.gene2.mentions[0]};{self.gene2.identifier}\t{self.variant1.mentions[0]};{self.variant1.identifier}\t{self.variant2.mentions[0]};{self.variant2.identifier}\t{self.text}"
    
    def __repr__(self):
        return str(self)
    
    def check(self, text):
        """
        Check that the sentences containing the entities all together have less than MAX_TOKENS tokens
        """
        min_idx = np.inf
        max_idx = -np.inf
        list_sentences = list(self.nlp(text).sents)
        for tag in self.ENTITY_TAGS:
            
            not_found = True
            idx = -1
            while not_found and idx < len(list_sentences) - 1:
                idx += 1
                if tag in list_sentences[idx].text:
                    not_found = False
                
            if idx < min_idx:
                min_idx = idx
            
            not_found = True
            idx = len(list_sentences)
            while not_found and idx > 0:
                idx -= 1
                if tag in list_sentences[idx].text:
                    not_found = False
            
            if idx > max_idx:
                max_idx = idx
        
        
        start, end = list_sentences[min_idx].start_char, list_sentences[max_idx].end_char
        text = text[start:end]
        tokens = self.generate_tokens(text)
        return text, tokens, len(tokens)
    
class Document(object):
    nlp = nlp
    MAX_TOKENS = MAX_DISTANCE
    ANNOTATION_DICT = ANNOTATION_DICT
    
    def __init__(self,pmcid, text, annotations):
        self.pmcid = pmcid
        self.text = text
        self.tokens = [token.idx for token in nlp(text)]
        self.genes, self.variants = self.split_annotations(annotations)
        self.instances = []
    
    def __str__(self):
        return f"Document {self.pmcid} with {len(self.genes)} different genes and {len(self.variants)} different variants and {len(self.instances)} valid instances"
    
    def __repr__(self):
        return str(self)
    
    @staticmethod
    def find_index(array, idx):
 
        start = 0
        end = len(array) - 1
        while start<= end:
 
            mid =(start + end)//2
 
            if array[mid] == idx:
                return mid
 
            elif array[mid] < idx:
                start = mid + 1
            else:
                end = mid-1
 
        return end
    
    def check_distance(self, entities):
        entities = sorted(entities)
        idx1, idx2 = [self.find_index(self.tokens,entity) for entity in entities]
        return (idx2 - idx1) > self.MAX_TOKENS
        
    
    @staticmethod
    def split_annotations(annotations):
        genes = []
        variants = []
        for annotation in annotations:
            if annotation.type=="Gene":
                genes.append(annotation)
            else:
                variants.append(annotation)
        return genes, variants
    
    def generate_groups(self):
        # select for each a couple of (gene,gene) and (variant, variant)
        groups = []
        for i in range(len(self.genes)-1):
            for gene1 in self.genes[i]:
                for j in range(i+1,len(self.genes)):
                    for gene2 in self.genes[j]:
                        if self.check_distance([gene1.get_start(),gene2.get_start()]):
                            continue
                        for k in range(len(self.variants)-1):
                            for variant1 in self.variants[k]:
                                if any([self.check_distance([gene1.get_start(),variant1.get_start()]), self.check_distance([gene2.get_start(),variant1.get_start()])]):
                                    continue
                                for l in range(k+1, len(self.variants)):
                                    for variant2 in self.variants[l]:
                                        if any([self.check_distance([gene1.get_start(),variant2.get_start()]), self.check_distance([gene2.get_start(),variant2.get_start()]), self.check_distance([variant1.get_start(),variant2.get_start()])]):
                                            continue
                                        groups.append((gene1,gene2,variant1,variant2))
        print(f"Found {len(groups)} groups of (gene, gene, variant, variant)")
        return groups
    
    def create_tagged_text(self,group):
        group = sorted(group)
        new_text = ''
        old_start = None
        for entity in group:
            start, end = entity.get_positions()
            if old_start is None:
                new_text += self.text[:start]
            else:
                new_text += self.text[old_start:start]
            new_text += self.ANNOTATION_DICT[entity.type]
            old_start = end
        new_text += self.text[old_start:]
        instance = OligoInstance(self.pmcid, group, new_text)
            
        if instance.is_valid():
            GLOBALLOCK.acquire()
            self.write_instances(f"{OUTPUT_DIRECTORY}{self.pmcid}.tsv", instance)
            GLOBALLOCK.release()
        return True
    
    def generate_instances(self):
        groups = self.generate_groups()
        with closing(mp.Pool(15, maxtasksperchild = 1000)) as p:
            max_ = len(groups)
            with tqdm(total=max_,desc='Combinations of entities') as pbar:
                for _ in p.imap_unordered(self.create_tagged_text, groups):
                    pbar.update()
        
    
    def write_instances(self, file, instance):
        with open(file, 'a') as outfile:
            if instance is not None:
                outfile.write(str(instance)+"\n")

def create_instances(document):
    annotations = []
    text = ""
    total_offset = 0
    for i, passage in enumerate(document['passages']):
        if total_offset != passage['offset']:
            text += " "*(passage['offset']-total_offset)
        text += passage['text']
        total_offset = len(text)
        for annotation in passage['annotations']:
            entity_type = annotation['infons']['type']
            try:
                identifier = annotation['infons']['NCBI Homologene'] if 'NCBI Homologene' in annotation['infons'] else annotation['infons']['Identifier']
            except:
                identifier = annotation['infons']['identifier']
            mention = annotation['text']
            for location in annotation['locations']:
                start = location['offset']
                end = start + location['length']
                annotations.append((entity_type, mention, identifier, (start, end)))
    annotations = sorted(annotations, key=lambda annot: annot[3])
    new_annotations = []
    for entity_type, mention, identifier, location in annotations:
        entity = Entity(entity_type, mention, identifier, location)
        try:
            idx = new_annotations.index(entity)
        except ValueError:
            idx = None
        if idx is not None:
            new_annotations[idx].merge(entity)
        else:
            new_annotations.append(entity)
    doc = Document(document['id'], text, new_annotations)
    with open(f"{OUTPUT_DIRECTORY}{document['id']}.tsv",'w') as outfile:
        outfile.write("pmcid\tgene1\tgene2\tvariant1\tvariant2\tsentence\n")
    doc.generate_instances()
    #doc.write_instances(f"{OUTPUT_DIRECTORY}{document['id']}.tsv")
    
if __name__ == "__main__":
    for i in tqdm(range(len(data['documents'])), desc="Files annotated"):
        create_instances(data['documents'][i])

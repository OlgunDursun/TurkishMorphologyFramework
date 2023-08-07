# %%
import pandas as pd
import csv
from collections import deque
import string
from .affix import Affix
from .word import Word




class Analyzer:

    def __init__(self):
        self


    def variant_checker_dictionary(self, affix_set, stem, word):
        return any(stem + affix == word[:len(stem + affix)] for affix in affix_set)

    def correct_variant_returner(self, affix_set, stem, word):
        return next((affix for affix in affix_set if stem + affix == word[:len(stem + affix)]), None)



    def rule1(self, word_object):
        "checking for three consecutive single-letter suffixes"
        word_as_list = word_object.deep_form
        for i in range(len(word_as_list)-2):
            if len(word_as_list[i]) == 1 and len(word_as_list[i+1]) == 1 and len(word_as_list[i+2]) == 1:
                return False
        return True


    def rule2(self, word_object, affixes, exception_list = None):
        "checking for inflectional suffixes before derivational suffixes"
        exception_list = ["DER087", "DER122", "DER141", "DER075", "DER076"]
        inflectional_found = False
        for suffix in word_object.suffixes:
            if suffix is None or suffix not in affixes:
                continue
            if affixes[suffix].functional_type == "inflectional":
                inflectional_found = True
            elif inflectional_found and affixes[suffix].functional_type == "derivational" and suffix not in exception_list:
                return False
        return True

    def rule3(self, word_object, unbreakable_roots = "unbreakable_roots.txt"):
        "checking for unbreakable roots"
        with open(unbreakable_roots, "r", encoding = "utf-8") as f:
            unbreakable_roots = f.read().splitlines()
        if word_object.root in unbreakable_roots:
            return True
        return False

    def rule4(self, word_object, forbidden_combinations):
        "checking for forbidden combinations"
        if len(word_object.suffixes) == 1:
            return True
        for i in range(len(word_object.suffixes)-1):
            if (word_object.suffixes[i], word_object.suffixes[i+1]) in forbidden_combinations:
                return False
        return True

    def rule5(self, word_object, overrides):
        "checking for overriding segmentations"



        deep_form_segments = word_object.deep_form
        deep_form_string = ''.join(deep_form_segments)

        if deep_form_string not in overrides:
            return True

        if deep_form_string in overrides and deep_form_segments == overrides[deep_form_string]:
            return True

        for i in range(len(deep_form_segments) - 1, -1, -1):
            sliced_string = ''.join(deep_form_segments[:i + 1])
            if sliced_string in overrides:
                if deep_form_segments[:i + 1] == overrides[sliced_string]:
                    return True
                else:
                    return False


    def rule6(self, word_object, affixes):
        """checking if affixes that have the peculiarity TAM1, TAM2, TAM3, TAM4, TAM5
        are in the correct order"""
        max_tam = 0
        for suffix in word_object.suffixes:
            if suffix is None:
                continue
            if str(affixes[suffix].peculiarity).startswith("TAM"):
                tam = int(affixes[suffix].peculiarity[3])
                if tam > max_tam:
                    max_tam = tam
                else:
                    print("internal",suffix)
                    return False
        return True


    def remove_duplicates(self, word_list):
        unique_words = {}
        for word in word_list:
            deep_form_key = (tuple(word.deep_form), word.pos, frozenset(word.morph_features.items()))
            if deep_form_key not in unique_words:
                unique_words[deep_form_key] = word
        return list(unique_words.values())


    def constraints(self, hyp, constraint_dict, affixes):
        if constraint_dict["ONLY_INFLECTIONAL"]:
            if not all(affixes[suffix].functional_type == "inflectional" for suffix in hyp.suffixes):
                return False
        
        if constraint_dict["ONLY_INFLECTIONAL"]:
            if not all(affixes[suffix].functional_type == "inflectional" for suffix in hyp.suffixes):
                return False

        return True


    def analyze(self, word, roots, affixes, forbidden_combinations, overrides):
        constraint_dict = {
                        "ONLY_INFLECTIONAL" : False,
                        "PROPER_APOSTROPHE" : False
                        }
        

        if word.isdigit():
            return [Word.generate_from_root(surface_input=word, root_input=word, root_pos="number", root_morph_features="NumType=Card")]

        elif word[0] == "\'":
            root_hyps = [Word.generate_from_root(surface_input="\'", root_input="\'", root_pos="noun", root_morph_features=None)]
            
        elif "\'" in word:
            proper_part = word.split("\'")[0]
            root_hyps = [Word.generate_from_root(surface_input=proper_part + "\'", root_input=proper_part, root_pos="noun", root_morph_features=None)]
            constraint_dict["PROPER_APOSTROPHE"] = True
            
        elif all(c in string.punctuation for c in word):
            return [Word.generate_from_root(surface_input=word, root_input=word, root_pos="punctuation", root_morph_features=None)]


        else:
            root_hyps = [
            hyp
            for root_name, root_data_list in roots.items() if word.startswith(root_name)
            for root_data in root_data_list
            for hyp in [
            Word.generate_from_root(surface_input=root_name, root_input=root_name, root_pos=root_data['type_en'], root_morph_features=root_data['morph_features'], semitic_root = root_data['semitic_root'], semitic_meter = root_data['semitic_meter']),
            *([Word.generate_from_root(surface_input=root_name[:-1] + root_data['suffixation'][0], root_input=root_name, root_pos=root_data['type_en'], root_morph_features=root_data['morph_features'], semitic_root = root_data['semitic_root'], semitic_meter = root_data['semitic_meter'])] if root_data['suffixation'] else [])
            ]
            ]
        queue = deque(root_hyps)
        visited = set()
        while queue:
            hyp = queue.popleft()
            if hyp in visited:
                continue
            if not self.rule1(hyp):  # check for rule1 compliance
                continue
            if not self.rule2(hyp, affixes):
                continue
            if not self.rule4(hyp, forbidden_combinations):
                continue
            #if not rule6(hyp, affixes):
            #    continue
            visited.add(hyp)
            fitting_keys = {key: value for key, value in affixes.items()
                            if self.variant_checker_dictionary(set(affixes[key].variants), stem=hyp.surface_form, word=word) and hyp.pos in affixes[key].input_pos}

            hypotheses = [obj for key, value in fitting_keys.items()
                        for obj in Word.generate_from_dict_hypothesis(hyp, value, (key, self.correct_variant_returner(set(value.variants), stem=hyp.surface_form, word=word)), affixes[key].representation)]


            #for item in hypotheses:
            #    item.update_pos_and_morph_features(affixes)

            queue.extend(hypotheses)

        out = [hyp for hyp in list(visited) if len(hyp.surface_form) == len(word) and self.rule5(hyp, overrides) and self.constraints(hyp, constraint_dict, affixes)]

        return (out)




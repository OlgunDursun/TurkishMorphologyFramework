

import pandas as pd
import csv
from collections import deque
import string
from .affix import Affix

class Word:
    def __init__(self, surface_form, deep_form, prefix, root, stem, suffixes, morph_features, pos = "noun"):
        self.surface_form = surface_form
        if type(deep_form) != list:
            self.deep_form = [deep_form]
        else:
            self.deep_form = deep_form
        self.prefix = prefix
        self.root = root
        self.stem = stem
        if type(suffixes) != list:
            self.suffixes = [suffixes]
        else:
            self.suffixes = suffixes
            
        self.morph_features = morph_features
        self.pos = pos

    def update_pos_and_morph_features(self, affixes):
        POS_FEATURES_TO_REMOVE = {
        "noun": ["Polarity", "Mood", "Tense"],
        "verb": ["Case"],
        "adjective": ["Polarity", "Mood", "Tense", "Case"],
        "adverb": [],
        "number": [],
        "pronoun": [],
        }
        
        POS_FEATURES_TO_ADD = {
        "noun": {"Case":"Nom","Number":"Sing","Person":"3"},
        "verb": {"Polarity":"Pos","Person":"3","Number":"Sing"},
        "adjective": [],
        "adverb": [],
        "number": [],
        "pronoun": [],
        }
        
        if self.suffixes:
            if self.suffixes[-1] in affixes:
                last_suffix = affixes[self.suffixes[-1]]
                previous_pos = self.pos
                self.pos = last_suffix.output_pos
            else:
                previous_pos = self.pos

            # Remove certain features if the part of speech has changed
            if previous_pos != self.pos:
                features_to_remove = POS_FEATURES_TO_REMOVE.get(previous_pos, []).copy()
                for feature in features_to_remove:
                    self.morph_features.pop(feature, None)

                # Add certain features if the part of speech has changed
                features_to_add = POS_FEATURES_TO_ADD.get(last_suffix.output_pos[0], {}).copy()  # add the .copy() method here
                self.morph_features.update(features_to_add)

        suffix_indices = {}  # add this dictionary to keep track of suffix indices
        for i, suffix in enumerate(self.suffixes):
            if suffix is None or suffix not in affixes:
                continue

            # Delete certain features if the suffix has requested their removal
            for key, value in affixes[suffix].wipe_features.items():
                if value == "*":
                    self.morph_features = {k: v for k, v in self.morph_features.items() if k != key}
                else:
                    self.morph_features.pop(key, None)

            for key, value in affixes[suffix].output_features.items():
                if key in self.morph_features:
                    # Check exceptions
                    # If a key already exists, update its value only if the current suffix has a higher index
                    if key not in suffix_indices or i > suffix_indices[key]:  # add this condition
                        self.morph_features[key] = value
                        suffix_indices[key] = i  # update the suffix index
                else:
                    self.morph_features[key] = value
                    suffix_indices[key] = i  # store the suffix index
        return self




    @classmethod
    def generate_from_dict_hypothesis(cls, previous_Word_object, dict_row, correct_variant, representation):
        deep_form = previous_Word_object.deep_form.copy()
        suffixes = previous_Word_object.suffixes.copy()

        # deep_form.append(representation) # uncomment this for another setting
        deep_form.append(correct_variant[1])
        suffixes.append(dict_row.affix_id)

        objects = []
        for pos in dict_row.output_pos:
            objects.append(cls(previous_Word_object.surface_form + correct_variant[1],                  #surface
                            deep_form.copy(),
                            None,                                                                       #prefix
                            previous_Word_object.root,                                                  #root
                            previous_Word_object.root,                                                  #stem
                            suffixes.copy(),
                            previous_Word_object.morph_features.copy(),                                        #morph_features
                            pos.strip()))                                                               #pos
        #print(objects)
        return objects




    @classmethod
    def generate_from_root(cls, surface_input, root_input, root_pos, root_morph_features, semitic_root = None, semitic_meter = None):
        DEFAULT_POS_FEATURES = {
            "noun": {"Case":"Nom","Number":"Sing","Person":"3"},
            "verb": {"Polarity":"Pos","Person":"3","Number":"Sing","Tense":"Pres", "Mood":"Imp"},
            "adjective": {},
            "pronoun": {},
        }
        
        if root_morph_features != None and len(root_morph_features) > 0:
            features = {}
            for feat in root_morph_features.split("|"):
                key_value = feat.split("=")
                if len(key_value) != 2:
                    print(f"Warning: Feature {feat} is not in the correct format.")
                    continue
                features[key_value[0]] = key_value[1]
            root_morph_features = features
        elif root_pos in DEFAULT_POS_FEATURES:
            root_morph_features = DEFAULT_POS_FEATURES[root_pos].copy()
        else:
            root_morph_features = {}
        if semitic_root and semitic_meter:
            return cls(surface_input, [semitic_root, semitic_meter], None, semitic_root, root_input, [semitic_meter], root_morph_features, root_pos)
        else:
            return cls(surface_input, root_input, None, root_input, root_input, [], root_morph_features, root_pos)
            

    def __hash__(self):
        return hash(tuple(self.deep_form))

    def __eq__(self, other):
        return self.root==other.root\
               and self.suffixes==other.suffixes\
                   and self.pos==other.pos
    def __repr__(self):
       return str(self.deep_form)
# %%
import pandas as pd
import csv
from collections import deque
import string
    

class Affix:
    def __init__(self, affix_id, representation, variants, input_pos, output_pos, 
                 input_features, output_features, wipe_features, positional_type, 
                 functional_type, peculiarity, example, metadata):
        self.affix_id = affix_id
        self.representation = representation
        self.variants = variants
        self.input_pos = input_pos
        self.output_pos = output_pos
        self.input_features = self.parse_features(input_features)
        self.output_features = self.parse_features(output_features)
        self.wipe_features = self.parse_features(wipe_features)
        self.positional_type = positional_type
        self.functional_type = functional_type
        self.peculiarity = peculiarity
        self.example = example
        self.metadata = metadata
    


    def parse_features(self, s):
        if not isinstance(s, str) or s == "nan":
            return {}
        features = {}
        for feat in s.split("|"):
            key_value = feat.split("=")
            if len(key_value) != 2:
                print(f"Warning: Feature {feat} is not in the correct format.")
                continue
            features[key_value[0]] = key_value[1]
        return features

    def __repr__(self):
       return str("+{}{}".format(self.representation, self.output_features))

    def __str__(self):
       return str(self.affix_id + " +" + self.representation)

    def __hash__(self):
        return hash(self.affix_id)

    def __eq__(self, other): 
        if not isinstance(other, Affix):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.affix_id == other.affix_id



# %%

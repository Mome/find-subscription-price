import time
import cmd
from pprint import pprint

import pandas as pd
from textblob import TextBlob


synonyms = {
    "smartphone" : "phone",
    "don't" : "do not",
    "doesn't" : "does not",
}

def replace_synonyms(text):
    words = (synonyms.get(w, w) for w in text.words)
    return TextBlob(' '.join(words))

def normalize(text):
    text = text.lower()
    text = replace_synonyms(text)
    #text = text.correct()    # spelling correction
    return text


class PreferenceModel:
    def __init__(self, data):
        self.cols = data.columns.tolist()
        self.brands = list(set(data['Brand']))
        self.categories = list(set(data['Category']))
        prices = data['Subscription Plan']
        self.price_range = (prices.min(), prices.max())
        
        self.price_pref = [None, None]
        self.category_pref = None
        self.brand_pref = [0.5]*len(self.categories)
    
    
    def __str__(self):
        return (
            f"Category: {self.category_pref}\n"
            f"Brand: {self.brand_pref}\n"
            f"Price Range: {self.price_pref}")


# ------------------------------------------------------------------------------ #


class ChatbotShell(cmd.Cmd):
    
    def __init__(self, data, model):
        super().__init__()
        self.prompt = '» '
        self.data = data
        self.model = model
    
    def precmd(self, line):
        if line.startswith(':'):
            return line[1:]
        else:
            return 'say ' + line
    
    def do_say(self, line):
        print('said:', line)
    
    def do_exit(self, line):
        return True
    
    def do_get(self, line):
        """Searches shell, model or globals and print first var with given name."""
        line = line.replace(' ', '_')
        for vars_ in [self, self.model, globals()]:
            if not isinstance(vars_, dict):
                vars_ = vars(vars_)
            if line in vars_:
                print(vars_[line])
                break
        else:
            print('cannont find identifier:', line)
        
    def default(self, line):
        self.do_get(line)


if __name__ == '__main__':
    data = pd.read_table('data.csv', index_col=0)
    model = PreferenceModel(data)
    cbs = ChatbotShell(data, model)
    cbs.cmdloop()
